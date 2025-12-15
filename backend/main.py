from fastapi import FastAPI, UploadFile, File, Depends
from backend.processing import process_comments_batch, process_single_comment
import pandas as pd
from backend.db import get_db, Comment
from sqlalchemy.orm import Session
from fastapi.responses import JSONResponse
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()


@app.post("/upload")
@app.post("/upload/")  # ✅ allow both with and without slash
async def upload_comments(file: UploadFile = File(...)):
    logger.debug("Received file upload: %s", file.filename)
    try:
        if file.filename.endswith(".csv"):
            df = pd.read_csv(file.file)
        else:
            df = pd.read_excel(file.file)

        logger.debug("File read successfully, rows: %d", len(df))
        processed_data = process_comments_batch(df)
        logger.debug("Processed %d comments", len(processed_data))

        db: Session = next(get_db())
        for item in processed_data:
            comment = Comment(**item)
            db.add(comment)
        db.commit()
        logger.debug("Data committed to database")

        return {"status": "processed", "count": len(processed_data)}
    except Exception as e:
        logger.error("Error processing upload: %s", str(e))
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/analyze")
@app.post("/analyze/")  # ✅ allow both
async def analyze_text(data: dict, db: Session = Depends(get_db)):
    logger.debug("Received text analysis request: %s", data)
    try:
        comment = data.get("comment")
        language = data.get("language", "en")
        if not comment:
            return JSONResponse(status_code=400, content={"error": "Comment is required"})

        processed_data = process_single_comment(
            comment=comment,
            language=language,
            section=data.get("section", "Unknown"),
            draft_version=data.get("draft_version", "v1"),
            date=data.get("date", "Unknown"),
            stakeholder=data.get("stakeholder", ""),
        )
        logger.debug("Processed single comment: %s", processed_data)

        comment_obj = Comment(**processed_data)
        db.add(comment_obj)
        db.commit()

        return {
            "sentiment": processed_data["sentiment"],
            "confidence": processed_data["confidence"],
            "summary": processed_data["summary"],
            "keywords": processed_data["keywords"],
            "priority": processed_data["priority"],
        }
    except Exception as e:
        logger.error("Error in text analysis: %s", str(e))
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/comments")
@app.get("/comments/")  # ✅ allow both
def get_analysis(draft_version: str = None, section: str = None, db: Session = Depends(get_db)):
    logger.debug("Received analysis request: draft_version=%s, section=%s", draft_version, section)
    try:
        query = db.query(Comment)
        if draft_version:
            query = query.filter(Comment.draft_version == draft_version)
        if section:
            query = query.filter(Comment.section == section)

        results = [c.__dict__ for c in query.all()]
        for r in results:
            r.pop("_sa_instance_state", None)

        logger.debug("Returning %d results", len(results))
        return JSONResponse(content=results)
    except Exception as e:
        logger.error("Error in analysis endpoint: %s", str(e))
        return JSONResponse(status_code=500, content={"error": str(e)})
