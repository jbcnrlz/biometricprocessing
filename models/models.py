from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship

Base = declarative_base()

class Database(Base):
    __tablename__ = 'Database'

    id = Column(Integer, primary_key=True)
    nome = Column(String)
    url_disp = Column(String)    

    def __repr__(self):
        return "<Database (nome='%s', url='%s')>" % (self.nome, self.url_disp)

class Metodo(Base):
    __tablename__ = 'Metodo'

    id = Column(Integer, primary_key=True)
    nome = Column(String)    

    def __repr__(self):
        return "<Metodo (nome='%s')>" % (self.nome)

class Sujeito(Base):
    __tablename__ = 'Sujeito'

    id = Column(Integer, primary_key=True)
    nome = Column(String)
    Database_id = Column(Integer, ForeignKey('Database.id'))    

    database = relationship("Database", back_populates="sujeitos")    

    def __repr__(self):
        return "<Database (nome='%s')>" % (self.nome)

class Template(Base):
    __tablename__ = 'Template'

    id = Column(Integer, primary_key=True)
    amostra = Column(String)
    caracteristica = Column(String)
    data_extracao = Column(DateTime)
    Sujeito_id = Column(Integer, ForeignKey('Sujeito.id'))
    Metodo_id = Column(Integer, ForeignKey('Metodo.id'))

    metodo = relationship("Metodo", back_populates="templates")    
    sujeito = relationship("Sujeito",back_populates="templates")

    def __repr__(self):
        return "<Database (amostra='%s')>" % (self.nome)


Database.sujeitos = relationship("Sujeito", order_by=Sujeito.nome, back_populates="database")
Metodo.templates = relationship("Template", order_by=Template.amostra, back_populates="metodo")
Sujeito.templates = relationship("Template",order_by=Template.amostra,back_populates="sujeito")