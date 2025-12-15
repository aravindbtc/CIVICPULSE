from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy_utils import ScalarListType
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class Comment(Base):
    __tablename__ = 'comments'
    id = Column(Integer, primary_key=True)
    original_comment = Column(String)
    translated_comment = Column(String)
    sentiment = Column(String)
    confidence = Column(Float)
    summary = Column(String)
    keywords = Column(ScalarListType())  # Array support
    section = Column(String)
    priority = Column(String, default="Normal")
    draft_version = Column(String)
    date = Column(String)
    stakeholder = Column(String)
    embedding = Column(ScalarListType())  # Array support
    cluster = Column(Integer)

engine = create_engine('sqlite:///db/database.db')
Base.metadata.create_all(engine)  # This recreates if table exists
Session = sessionmaker(bind=engine)

def get_db():
    db = Session()
    try:
        yield db
    finally:
        db.close()