import uuid
from datetime import datetime
from sqlalchemy import Column, String, Integer, BigInteger, Numeric, Text, DateTime, ForeignKey, ARRAY
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from app.core.database import Base

def new_uuid(): return str(uuid.uuid4())

class Dataset(Base):
    __tablename__ = "datasets"
    id         = Column(Text, primary_key=True, default=new_uuid)
    name       = Column(String(255), nullable=False)
    source     = Column(String(50), nullable=False, default="upload")
    file_path  = Column(Text)
    file_size  = Column(BigInteger)
    status     = Column(String(20), nullable=False, default="ready")
    metadata_  = Column("metadata", JSONB, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    pipeline_runs = relationship("PipelineRun", back_populates="dataset", lazy="selectin")

class PipelineRun(Base):
    __tablename__ = "pipeline_runs"
    id           = Column(Text, primary_key=True, default=new_uuid)
    dataset_id   = Column(Text, ForeignKey("datasets.id"), nullable=False)
    content_hash = Column(Text)
    status       = Column(String(20), default="running")
    result       = Column(JSONB, default=dict)
    created_at   = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    dataset      = relationship("Dataset", back_populates="pipeline_runs")
