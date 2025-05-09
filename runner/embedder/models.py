# models.py
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Metadata(Base):
    __tablename__ = 'metadata'
    
    source_url = Column(String, primary_key=True)
    content_type = Column(String)
    # using last_mod as string
    last_modified = Column(String)
    fetch_timestamp = Column(DateTime)
    raw_filepath = Column(String)


class Content(Base):
    __tablename__ = 'content'

    id = Column(Integer, primary_key=True, autoincrement=True)
    source_path = Column(String)
    content_type = Column(String)
    last_modified = Column(DateTime)
    content = Column(String)
    
class Embedded(Base):
    __tablename__ = 'embedded'

    id = Column(Integer, autoincrement=True, primary_key=True)
    content_id = Column(Integer, ForeignKey('content.id'))
    cleaned_content = Column(String)
    last_updated = Column(DateTime)