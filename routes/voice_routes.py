from llm.groq_llm import get_groq_reply, clean_ai_response
from llm.fast_rag import get_fast_rag_response
import os
from dotenv import load_dotenv
from fastapi import Request, Response, APIRouter
from twilio.twiml.voice_response import VoiceResponse, Gather
from twilio.rest import Client
import time
from functools import lru_cache

load_dotenv()

router = APIRouter()

# Read environment variables
TWILIO_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_CALLER_ID = os.getenv("TWILIO_CALLER_ID")
TWILIO_CALLBACK_URL = os.getenv("TWILIO_CALLBACK_URL")

client = Client(TWILIO_SID, TWILIO_AUTH)

# Cache for common responses
@lru_cache(maxsize=100)
def get_cached_rag_response(question: str) -> str:
    """Get cached RAG response for common questions."""
    result = get_fast_rag_response(question)
    if isinstance(result, tuple):
        return result[0]  # Return just the answer, not the timings
    return result

def escape_xml(text):
    """Escape characters for XML compliance."""
    return (
        text.replace('&', '&amp;')
            .replace('<', '&lt;')
            .replace('>', '&gt;')
            .replace('"', '&quot;')
            .replace("'", '&apos;')
    )

@router.api_route("/", methods=["GET", "POST"])
async def root():
    return {"message": "Technology Mindz Virtual Manager API is running"}

@router.api_route("/gather-speech", methods=["GET", "POST"])
async def gather_speech(request: Request):
    start_time = time.time()
    form = await request.form()
    after_form_time = time.time()
    speech_result = str(form.get("SpeechResult", ""))
    speech_result_alt = str(form.get("speech_result", ""))
    confidence = str(form.get("Confidence", "0"))
    confidence_alt = str(form.get("confidence", "0"))
    attempt_count = int(str(form.get("attempt_count", "0")))
    interruption_count = int(str(form.get("interruption_count", "0")))
    
    if not speech_result and speech_result_alt:
        speech_result = speech_result_alt
    if confidence == "0" and confidence_alt != "0":
        confidence = confidence_alt
    
    if not speech_result:
        response = VoiceResponse()
        
        if attempt_count == 0:
            # First attempt - professional greeting
            response.say("Hello, this is Roney, your virtual assistant at Technology Mindz. I'm here to help you with our software development, AI, and Salesforce services. Please tell me how I can assist you today.")
        elif attempt_count == 1:
            # Second attempt - gentle reminder
            response.say("I didn't catch that. Please speak clearly and let me know what you need help with. You can ask about our services, contact information, or any questions about Technology Mindz.")
        elif attempt_count == 2:
            # Third attempt - alternative options
            response.say("I'm having trouble hearing you. You can also reach us at +1 (888) 982-4016 or email us at info@technologymindz.com for immediate assistance. Thank you for calling Technology Mindz.")
            response.hangup()
            print(f"[CALL TIMING] start={start_time:.3f} after_form={after_form_time:.3f} total={time.time()-start_time:.3f} [NO SPEECH - FINAL]")
            return Response(content=str(response), media_type="application/xml")
        else:
            # Fallback - professional goodbye
            response.say("Thank you for calling Technology Mindz. You can reach us at +1 (888) 982-4016 or visit our website for more information. Have a great day!")
            response.hangup()
            print(f"[CALL TIMING] start={start_time:.3f} after_form={after_form_time:.3f} total={time.time()-start_time:.3f} [NO SPEECH - EXIT]")
            return Response(content=str(response), media_type="application/xml")
        
        # Increment attempt count for next try
        next_attempt = attempt_count + 1
        gather = Gather(
            input='speech',
            action=f'{TWILIO_CALLBACK_URL}/gather-speech?attempt_count={next_attempt}&interruption_count=0',
            method='POST',
            speech_timeout='auto',
            language='en-US',
            speech_model='phone_call',
            enhanced=True
        )
        response.append(gather)
        
        # Professional timeout message
        if attempt_count == 0:
            response.say("I'm listening. Please go ahead with your question.")
        else:
            response.say("I'm still here and ready to help. Please speak now.")
        
        print(f"[CALL TIMING] start={start_time:.3f} after_form={after_form_time:.3f} total={time.time()-start_time:.3f} [NO SPEECH - ATTEMPT {attempt_count + 1}]")
        return Response(content=str(response), media_type="application/xml")
    
    try:
        speech_text = str(speech_result).strip()
        confidence_float = float(confidence)
        
        if confidence_float < 0.2:
            response = VoiceResponse()
            response.say("I'm sorry, I didn't catch that clearly. Could you please repeat your question?")
            gather = Gather(
                input='speech',
                action=f'{TWILIO_CALLBACK_URL}/gather-speech',
                method='POST',
                speech_timeout='auto',
                language='en-US',
                speech_model='phone_call',
                enhanced=True
            )
            gather.say("Please repeat your question.")
            response.append(gather)
            response.say("I still didn't hear anything. Please try again.")
            response.redirect(f'{TWILIO_CALLBACK_URL}/gather-speech')
            print(f"[CALL TIMING] start={start_time:.3f} after_form={after_form_time:.3f} total={time.time()-start_time:.3f} [LOW CONFIDENCE]")
            return Response(content=str(response), media_type="application/xml")
        
        # Optimized RAG Integration with timing
        before_rag_time = time.time()
        
        # Try cached response first for common questions
        common_questions = [
            "contact", "phone", "email", "reach", "call", "number"
        ]
        is_common_question = any(word in speech_text.lower() for word in common_questions)
        
        if is_common_question:
            rag_answer = get_cached_rag_response(speech_text)
            rag_timings = None
        else:
            result = get_fast_rag_response(speech_text, return_timings=True)
            if isinstance(result, tuple):
                rag_answer, rag_timings = result
            else:
                rag_answer = result
                rag_timings = None
        
        after_rag_time = time.time()
        
        # Clean the RAG response
        ai_reply = clean_ai_response(rag_answer)
        
        response = VoiceResponse()
        if not ai_reply.strip():
            ai_text = "I'm sorry, I didn't catch that. Could you please repeat your question?"
        else:
            ai_text = escape_xml(ai_reply)
        
        # Handle interruptions professionally
        if interruption_count > 0:
            # If user interrupted, acknowledge and continue
            response.say("I understand you'd like to speak. Please go ahead with your question.")
            gather = Gather(
                input='speech',
                action=f'{TWILIO_CALLBACK_URL}/gather-speech?interruption_count=0',
                method='POST',
                speech_timeout='auto',
                language='en-US',
                speech_model='phone_call',
                enhanced=True
            )
            response.append(gather)
            response.say("I'm listening. Please speak now.")
        else:
            # Normal response flow
            gather = Gather(
                input='speech',
                action=f'{TWILIO_CALLBACK_URL}/gather-speech?interruption_count=0',
                method='POST',
                speech_timeout='auto',
                language='en-US',
                speech_model='phone_call',
                enhanced=True,
                barge_in=True  # Allow interruption
            )
            gather.say(ai_text)
            response.append(gather)
            
            # Professional closing message
            response.say("Thank you for calling Technology Mindz. If you have more questions, please speak now, or you can call us directly at +1 (888) 982-4016. Have a great day!")
            response.hangup()
        
        end_time = time.time()
        total_rag_time = after_rag_time - before_rag_time
        total_call_time = end_time - start_time
        
        if is_common_question:
            print(f"[CALL RAG TIMING] speech='{speech_text[:50]}...' rag_total={total_rag_time:.3f}s call_total={total_call_time:.3f}s confidence={confidence_float:.2f} [CACHED]")
        else:
            if rag_timings:
                print(f"[CALL RAG TIMING] speech='{speech_text[:50]}...' retrieval={rag_timings.get('retrieval_time', 0):.3f}s generation={rag_timings.get('generation_time', 0):.3f}s rag_total={total_rag_time:.3f}s call_total={total_call_time:.3f}s confidence={confidence_float:.2f}")
            else:
                print(f"[CALL RAG TIMING] speech='{speech_text[:50]}...' rag_total={total_rag_time:.3f}s call_total={total_call_time:.3f}s confidence={confidence_float:.2f}")
        
        return Response(content=str(response), media_type="application/xml")
    except Exception as e:
        response = VoiceResponse()
        response.say("I'm sorry, I encountered an error. Please try again.")
        gather = Gather(
            input='speech',
            action=f'{TWILIO_CALLBACK_URL}/gather-speech',
            method='POST',
            speech_timeout='auto',
            language='en-US',
            speech_model='phone_call',
            enhanced=True
        )
        gather.say("Please try asking your question again.")
        response.append(gather)
        response.say("I didn't hear anything. Please try again.")
        response.redirect(f'{TWILIO_CALLBACK_URL}/gather-speech')
        print(f"[CALL TIMING] start={start_time:.3f} after_form={after_form_time:.3f} total={time.time()-start_time:.3f} [EXCEPTION: {str(e)}]")
        return Response(content=str(response), media_type="application/xml")

@router.post("/make-call")
async def make_call(request: Request):
    try:
        data = await request.json()
        phone_number = data["to"]

        if not phone_number.startswith('+'):
            phone_number = '+' + phone_number

        if not TWILIO_CALLER_ID:
            return {"error": "TWILIO_CALLER_ID not configured"}, 400

        # Create call with professional initial message
        call = client.calls.create(
            to=phone_number,
            from_=TWILIO_CALLER_ID,
            url=f"{TWILIO_CALLBACK_URL}/gather-speech?attempt_count=0",
            status_callback=f"{TWILIO_CALLBACK_URL}/call-status",
            status_callback_event=['completed'],
            status_callback_method='POST'
        )

        return {
            "sid": call.sid, 
            "status": "initiated",
            "message": "Professional call initiated with improved no-speech handling"
        }
    except Exception as e:
        return {"error": str(e)}, 400

@router.get("/call-status/{call_sid}")
async def get_call_status(call_sid: str):
    """
    Get the status of a specific call.
    """
    try:
        call = client.calls(call_sid).fetch()
        return {"sid": call.sid, "status": call.status}
    except Exception as e:
        return {"error": str(e)}, 400

@router.post("/rag-chat")
async def rag_chat(request: Request):
    """Fast RAG endpoint for real-time responses with latency logging."""
    try:
        data = await request.json()
        question = data.get("question", "")
        if not question:
            return {"error": "No question provided"}, 400
        start_time = time.time()
        result = get_fast_rag_response(question, return_timings=True)
        end_time = time.time()
        
        if isinstance(result, tuple):
            answer, timings = result
            timings["api_total_time"] = end_time - start_time
            print(f"[RAG TIMING] retrieval={timings.get('retrieval_time', 0):.3f}s, generation={timings.get('generation_time', 0):.3f}s, total={timings.get('total_time', 0):.3f}s, api_total={timings['api_total_time']:.3f}s")
            return {"answer": answer, "timings": timings}
        else:
            answer = result
            timings = {"api_total_time": end_time - start_time}
            print(f"[RAG TIMING] api_total={timings['api_total_time']:.3f}s")
            return {"answer": answer, "timings": timings}
    except Exception as e:
        return {"error": str(e)}, 500
