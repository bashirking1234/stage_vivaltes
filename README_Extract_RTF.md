# Extract RTF Data

Dit Python-script is ontworpen om specifieke informatie uit een RTF-bestand (Rich Text Format) te extraheren en te bewerken. 
Dit script is vooral handig voor wetenschappers en onderzoekers die werken met post-translationele modificaties (PTM's) en 
de atoomverbindingen binnen deze modificaties willen analyseren.

## Functies

- Lezen van RTF-bestanden.
- Extraheren van PTM-informatie uit de RTF-regels.
- Identificeren van atoomverbindingen, inclusief enkele en dubbele bindingen.
- Bijwerken van atoomtypes op basis van hun verbindingen.
- Verwijderen van waterstofatomen uit bindingen.
- Schrijven van de bewerkte informatie naar een uitvoerbestand.

## Bestanden in het project

  Er zijn meerdere RTF-bestanden beschikbaar in het project die gebruikt kunnen worden. 
  1. ARG.txt
  2. GLU.txt
  3. CIR.txt

  zelf kunt u ook andere bestanden installeren met atoomverbindengen. 



## Gebruik

1. Plaats je RTF-bestand in de projectdirectory.

2. Wijzig het `rtf_filepath` en `output_path` in de `main()` functie naar de juiste paden van jouw bestanden:

  
    rtf_filepath = "/pad/naar/jouw/RTF-bestand.txt"
    output_path = "/pad/naar/jouw/outputfile.txt"
  

## Codeoverzicht

### Functies

- `read_rtf_file(filepath)`: Leest de inhoud van een RTF-bestand.
- `write_to_file(filepath, content)`: Schrijft inhoud naar een opgegeven bestand.
- `extract_ptm_info(ptm, lines)`: Extraheert informatie voor een specifieke PTM uit de RTF-regels.
- `obtain_separate_rtf(ptm, filepath)`: Verkrijgt de RTF-informatie voor een specifieke PTM en schrijft deze naar aparte bestanden.
- `find_atom_connections(ptm)`: Vindt de atoomverbindingen in een PTM, identificerend enkele en dubbele bindingen.
- `update_atom_type(ptm, connections)`: Werkt het atoomtype bij op basis van verbindingen.
- `remove_hydrogen_from_bonds(ptm)`: Verwijdert waterstofatomen uit bindingen in de PTM.
- `write_to_output_file(ptm, output_path)`: Schrijft de bewerkte PTM-informatie naar een uitvoerbestand.

## Licentie
Python Licensing
This project uses Python, which is released under the Python Software Foundation (PSF) License. 
The PSF License is a permissive open-source license that allows you to freely use, modify, and distribute Python for any purpose, including commercial applications.

For more details, please refer to the (PSF License Agreement).

Extract_RTF_data is auteursrechtelijk beschermd in (2024) door de HAN University of Applied Sciences. Alle rechten voorbehouden.

## Auteur
Bashir Hussein

HAN University of Applied Sciences (2024)
