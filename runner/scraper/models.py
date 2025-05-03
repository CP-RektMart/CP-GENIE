# models.py
from sqlalchemy import Column, String, DateTime
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