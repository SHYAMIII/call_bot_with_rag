import asyncio
import json
import base64
from typing import Dict, Optional
from llm.groq_llm import get_groq_reply, clean_ai_response, get_short_system_prompt, get_context_system_prompt, remove_leading_greeting
from tts.streaming_tts import streaming_tts
import main

# Get the Socket.IO instance from main
sio = main.sio

# Store active conversations
active_conversations: Dict[str, Dict] = {}

@sio.event
async def start_conversation(sid, data):
    """Start a new conversation session"""
    active_conversations[sid] = {
        'context': [],
        'is_speaking': False,
        'interrupted': False,
        'current_task': None,
        'audio_queue': [],
        'processing': False,
        'greeted': False  # Track if greeting has been sent
    }
    await sio.emit('conversation_started', {'status': 'ready'}, room=sid)

    # --- Intro message streaming with interruption support ---
    intro_message = "Hello this is Roney, the virtual manager at Technology Mindz, How can I assist you ?"
    conversation = active_conversations[sid]
    conversation['is_speaking'] = True
    conversation['interrupted'] = False
    conversation['processing'] = False
    async def stream_intro():
        chunk_count = 0
        try:
            intro_chunks = await streaming_tts.split_text_for_streaming(intro_message, max_chunk_length=60)
            await sio.emit('ai_response_started', {'text': intro_message}, room=sid)
            for chunk in intro_chunks:
                if conversation['interrupted']:
                    conversation['is_speaking'] = False
                    conversation['processing'] = False
                    await sio.emit('interruption_handled', {'status': 'stopped'}, room=sid)
                    return
                async for audio_data in streaming_tts.generate_streaming_audio(chunk):
                    if conversation['interrupted']:
                        conversation['is_speaking'] = False
                        conversation['processing'] = False
                        await sio.emit('interruption_handled', {'status': 'stopped'}, room=sid)
                        return
                    if conversation['interrupted']:
                        conversation['is_speaking'] = False
                        conversation['processing'] = False
                        await sio.emit('interruption_handled', {'status': 'stopped'}, room=sid)
                        return
                    audio_b64 = base64.b64encode(audio_data).decode('utf-8')
                    await sio.emit('audio_chunk', {
                        'chunk_index': chunk_count,
                        'total_chunks': None,
                        'audio_data': audio_b64,
                        'text_chunk': chunk
                    }, room=sid)
                    chunk_count += 1
            conversation['is_speaking'] = False
            conversation['greeted'] = True  # Mark greeting as sent
            await sio.emit('response_complete', {'status': 'finished'}, room=sid)
        except asyncio.CancelledError:
            conversation['is_speaking'] = False
            conversation['processing'] = False
            await sio.emit('interruption_handled', {'status': 'stopped'}, room=sid)
        except Exception as e:
            conversation['is_speaking'] = False
            conversation['processing'] = False
            await sio.emit('error', {'message': 'Error during intro'}, room=sid)
    conversation['current_task'] = asyncio.create_task(stream_intro())

@sio.event
async def user_speech(sid, data):
    """Handle incoming user speech with streaming response"""
    if sid not in active_conversations:
        await start_conversation(sid, {})
    
    conversation = active_conversations[sid]
    
    # Check if AI is currently speaking (including intro) and handle interruption
    if conversation['is_speaking'] or conversation['processing']:
        conversation['interrupted'] = True
        conversation['is_speaking'] = False
        conversation['processing'] = False
        
        # Cancel current task if it exists (this will stop intro or regular response)
        if conversation['current_task'] and not conversation['current_task'].done():
            conversation['current_task'].cancel()
        
        await sio.emit('interruption_handled', {'status': 'stopped'}, room=sid)
    
    # Get user's speech text
    speech_text = data.get('text', '')
    if not speech_text:
        await sio.emit('error', {'message': 'No speech text received'}, room=sid)
        return
    
    # Add to conversation context
    conversation['context'].append({'role': 'user', 'content': speech_text})
    
    # Start processing in background
    conversation['processing'] = True
    conversation['interrupted'] = False
    conversation['current_task'] = asyncio.create_task(process_speech_with_streaming(sid, speech_text))

async def process_speech_with_streaming(sid: str, speech_text: str):
    """Process speech and stream response back to client with LLM streaming."""
    conversation = active_conversations[sid]
    try:
        await sio.emit('processing_started', {'status': 'thinking'}, room=sid)
        # Use context-aware system prompt after greeting
        if conversation.get('greeted', False):
            system_prompt = get_context_system_prompt()
        else:
            system_prompt = get_short_system_prompt()
        ai_response_full = ""
        chunk_buffer = ""
        chunk_count = 0
        async for llm_chunk in get_groq_reply(speech_text, system_prompt=system_prompt):
            if conversation['interrupted']:
                return
            ai_response_full += llm_chunk
            chunk_buffer += llm_chunk
            if len(chunk_buffer) > 30 or any(chunk_buffer.endswith(p) for p in ['.', '!', '?']):
                # Emit text chunk for browser-based TTS
                await sio.emit('text_chunk', {
                    'chunk_index': chunk_count,
                    'text_chunk': chunk_buffer
                }, room=sid)
                # Existing audio streaming logic
                async for audio_data in streaming_tts.generate_streaming_audio(chunk_buffer):
                    if conversation['interrupted']:
                        return
                    audio_b64 = base64.b64encode(audio_data).decode('utf-8')
                    await sio.emit('audio_chunk', {
                        'chunk_index': chunk_count,
                        'total_chunks': None,
                        'audio_data': audio_b64,
                        'text_chunk': chunk_buffer
                    }, room=sid)
                    chunk_count += 1
                chunk_buffer = ""
        if chunk_buffer:
            await sio.emit('text_chunk', {
                'chunk_index': chunk_count,
                'text_chunk': chunk_buffer
            }, room=sid)
            async for audio_data in streaming_tts.generate_streaming_audio(chunk_buffer):
                if conversation['interrupted']:
                    return
                audio_b64 = base64.b64encode(audio_data).decode('utf-8')
                await sio.emit('audio_chunk', {
                    'chunk_index': chunk_count,
                    'total_chunks': None,
                    'audio_data': audio_b64,
                    'text_chunk': chunk_buffer
                }, room=sid)
                chunk_count += 1
        ai_response_full = clean_ai_response(ai_response_full)
        if conversation.get('greeted', False):
            ai_response_full = remove_leading_greeting(ai_response_full)
        conversation['context'].append({'role': 'assistant', 'content': ai_response_full})
        conversation['is_speaking'] = False
        conversation['processing'] = False
        await sio.emit('ai_response_started', {'text': ai_response_full}, room=sid)
        await sio.emit('response_complete', {'status': 'finished'}, room=sid)
        conversation['greeted'] = True  # Mark greeting as sent after first response
    except asyncio.CancelledError:
        conversation['is_speaking'] = False
        conversation['processing'] = False
    except Exception as e:
        conversation['is_speaking'] = False
        conversation['processing'] = False
        await sio.emit('error', {'message': 'Error processing your request'}, room=sid)

@sio.event
async def interrupt(sid, data):
    """Handle user interruption during AI response"""
    if sid in active_conversations:
        conversation = active_conversations[sid]
        conversation['interrupted'] = True
        conversation['is_speaking'] = False
        conversation['processing'] = False
        
        # Cancel current task if it exists
        if conversation['current_task'] and not conversation['current_task'].done():
            conversation['current_task'].cancel()
        
        await sio.emit('interruption_handled', {'status': 'stopped'}, room=sid)

@sio.event
async def end_conversation(sid, data):
    """End conversation and clean up"""
    if sid in active_conversations:
        conversation = active_conversations[sid]
        
        # Cancel any ongoing tasks
        if conversation['current_task'] and not conversation['current_task'].done():
            conversation['current_task'].cancel()
        
        del active_conversations[sid]
        await sio.emit('conversation_ended', {'status': 'ended'}, room=sid)

@sio.event
async def get_conversation_status(sid, data):
    """Get current conversation status"""
    if sid in active_conversations:
        conversation = active_conversations[sid]
        await sio.emit('conversation_status', {
            'is_speaking': conversation['is_speaking'],
            'interrupted': conversation['interrupted'],
            'processing': conversation['processing'],
            'context_length': len(conversation['context'])
        }, room=sid)
    else:
        await sio.emit('conversation_status', {'error': 'No active conversation'}, room=sid) 