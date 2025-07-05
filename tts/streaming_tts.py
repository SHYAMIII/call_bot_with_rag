import edge_tts
from typing import AsyncGenerator

class StreamingTTS:
    def __init__(self, voice="en-US-GuyNeural"):
        self.voice = voice

    async def generate_streaming_audio(self, text: str) -> AsyncGenerator[bytes, None]:
        try:
            communicate = edge_tts.Communicate(text, self.voice)
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    yield chunk["data"]
        except Exception:
            yield b""

    async def split_text_for_streaming(self, text: str, max_chunk_length: int = 100) -> list:
        sentences = text.split('. ')
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < max_chunk_length:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

# Global instance
streaming_tts = StreamingTTS()
