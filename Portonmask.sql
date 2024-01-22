drop database if exists donneespatient;
create database if not exists donneespatient;
use donneespatient;

DROP TABLE IF EXISTS Rdv;
DROP TABLE IF EXISTS Patient;
DROP TABLE IF EXISTS Affichage;

CREATE TABLE Patient (
    FileVisage VARCHAR(500) PRIMARY KEY,
    Nom VARCHAR(50),
    Prenom VARCHAR(50),
    Ratios VARCHAR(1000)
);

CREATE TABLE Rdv (
    Id VARCHAR(50) PRIMARY KEY,
    Nom_Medecin VARCHAR(50),
    Prenom_Medecin VARCHAR(50),
    Emplacement VARCHAR(50),
    Daterdv DATE,
    Patient_FileVisage VARCHAR(500),
    Heure VARCHAR(10),
    FOREIGN KEY (Patient_FileVisage) REFERENCES Patient (FileVisage)
);

INSERT INTO Patient(FileVisage, Nom, Prenom, Ratios) VALUES
("Jacques.jpg", "Jacques", "Jacques", "[1,2.3,1,10]");

INSERT INTO Rdv (Id, Nom_Medecin, Prenom_Medecin, Emplacement, Daterdv, Heure, Patient_FileVisage)
VALUES ('RDV001', 'Dupont', 'Jean', 'Cabinet A', '2023-06-22', '12:00:00', 'Jacques.jpg');

INSERT INTO Rdv (Id, Nom_Medecin, Prenom_Medecin, Emplacement, Daterdv, Heure, Patient_FileVisage)
VALUES ('RDV002', 'Smith', 'Alice', 'Cabinet B', '2023-06-23', '12:30:00', 'Jacques.jpg');


SELECT * FROM Patient;
SELECT * FROM Rdv;


CREATE TABLE Affichage (
    FileVisage VARCHAR(500),
    Id VARCHAR(50),
    Nom VARCHAR(50),
    Prenom VARCHAR(50),
    Nom_Medecin VARCHAR(50),
    Emplacement VARCHAR(50),
    Daterdv DATE,
    Heure TIME,
    PRIMARY KEY (FileVisage, Id)
);

-- Insertion des données dans la nouvelle table en utilisant la jointure
INSERT INTO Affichage (Id,FileVisage, Nom, Prenom, Nom_Medecin, Emplacement, Daterdv,Heure)
SELECT Rdv.Id,Patient.FileVisage, Patient.Nom, Patient.Prenom, Rdv.Nom_Medecin, Rdv.Emplacement, Rdv.Daterdv, Rdv.Heure
FROM Patient
JOIN Rdv ON Patient.FileVisage = Rdv.Patient_FileVisage
ORDER BY Heure ASC;

SELECT * FROM Affichage;