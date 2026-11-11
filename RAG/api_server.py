import os
import uuid
import tempfile
import logging
import shutil
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage

from huflit_rag_agent import create_agent_graph
from huflit_voice_module import transcribe_audio, generate_speech

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="HUFLIT Robot Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.info("Khởi động Agent Graph...")
agent_app = create_agent_graph()

ROBOT_THREAD_ID = str(uuid.uuid4())

@app.post("/api/chat_voice")
async def chat_voice(audio: UploadFile = File(...)):
    global ROBOT_THREAD_ID
    
    temp_dir = tempfile.gettempdir()
    input_audio_path = os.path.join(temp_dir, f"input_{uuid.uuid4()}.webm")
    
    with open(input_audio_path, "wb") as buffer:
        shutil.copyfileobj(audio.file, buffer)
        
    logging.info(f"Đã nhận file Audio, dung lượng: {os.path.getsize(input_audio_path)} bytes")
    
    user_text = transcribe_audio(input_audio_path)
    
    try:
        os.remove(input_audio_path)
    except:
        pass
        
    if not user_text:
        return JSONResponse(content={"error": "Không nghe rõ âm thanh", "user_text": "", "ai_text": "", "audio_url": ""})
        
    logging.info(f"[User]: {user_text}")
    
    config = {"configurable": {"thread_id": ROBOT_THREAD_ID}}
    inputs = {"messages": [HumanMessage(content=user_text)]}
    
    final_output = ""
    for output in agent_app.stream(inputs, config=config, stream_mode="values"):
        if "generation" in output and output["generation"]:
            final_output = output["generation"]

    if not final_output.strip():
        logging.error("Agent trả về nội dung rỗng; sử dụng câu trả lời dự phòng.")
        final_output = (
            "Tôi chưa thể tạo câu trả lời vào lúc này. "
            "Bạn vui lòng hỏi lại bằng một câu đầy đủ hơn."
        )
                
    logging.info(f"[AI]: {final_output}")
    
    out_audio_path = generate_speech(final_output)
    
    audio_filename = f"response_{uuid.uuid4()}.wav"
    static_audio_dir = os.path.join(os.path.dirname(__file__), "frontend", "audio_cache")
    os.makedirs(static_audio_dir, exist_ok=True)
    
    public_audio_path = os.path.join(static_audio_dir, audio_filename)
    
    audio_url = ""
    if out_audio_path and os.path.exists(out_audio_path):
        shutil.move(out_audio_path, public_audio_path)
        audio_url = f"/audio_cache/{audio_filename}"
    
    return JSONResponse(content={
        "user_text": user_text,
        "ai_text": final_output,
        "audio_url": audio_url
    })

@app.post("/api/reset")
async def reset_chat():
    global ROBOT_THREAD_ID
    ROBOT_THREAD_ID = str(uuid.uuid4())
    return {"status": "ok", "message": "Đã reset bộ nhớ Robot."}

static_audio_dir = os.path.join(os.path.dirname(__file__), "frontend", "audio_cache")
os.makedirs(static_audio_dir, exist_ok=True)

app.mount("/audio_cache", StaticFiles(directory=static_audio_dir), name="audio_cache")
app.mount("/", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "frontend"), html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=False)
