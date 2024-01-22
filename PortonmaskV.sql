drop database if exists donneespatient;
create database if not exists donneespatient;
use donneespatient;


DROP TABLE IF EXISTS Patient;
CREATE TABLE Patient (
    FileVisage VARCHAR(500) PRIMARY KEY,
    Nom VARCHAR(50),
    Prenom VARCHAR(50),
    Ratios VARCHAR(1000)
);

DROP TABLE IF EXISTS Docteur;
CREATE TABLE Docteur (
    ID VARCHAR(50) PRIMARY KEY,
    Nom VARCHAR(50),
    Prenom VARCHAR(50),
    Fonction VARCHAR(100)
);

DROP TABLE IF EXISTS Rdv;
CREATE TABLE Rdv (
    Id VARCHAR(50) PRIMARY KEY,
    Emplacement VARCHAR(50),
    Daterdv DATE,
    Patient_FileVisage VARCHAR(500),
    Heure VARCHAR(10),
    ID_Doctor VARCHAR(50),
    FOREIGN KEY (ID_Doctor) REFERENCES Docteur (ID),
    FOREIGN KEY (Patient_FileVisage) REFERENCES Patient (FileVisage)
);

SELECT * FROM Patient;
SELECT * FROM Rdv ORDER BY Daterdv, Heure;
SELECT * FROM Docteur;