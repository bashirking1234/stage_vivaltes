#Automated_Regression Copyright (C) 2023 Bashir Hussein
import os

def read_rtf_file(filepath):
    """
    Leest de inhoud van een RTF-bestand.

    :param filepath: Pad naar het RTF-bestand.
    :return: Lijst van regels uit het bestand.
    """
    with open(filepath, "r") as file:
        lines = file.readlines()
        print(f"Read {len(lines)} lines from {filepath}")
        return lines

def write_to_file(filepath, content):
    """
    Schrijft inhoud naar een opgegeven bestand.

    :param filepath: Pad naar het bestand.
    :param content: Inhoud om naar het bestand te schrijven.
    """
    with open(filepath, "w") as file:
        file.write(content)
    print(f"Wrote content to {filepath}")

def extract_ptm_info(ptm, lines):
    """
    Extraheert informatie voor een specifieke PTM (post-translationele modificatie) uit de RTF-regels.

    :param ptm: Naam van de PTM.
    :param lines: Lijst van regels uit het RTF-bestand.
    :return: Geëxtraheerde informatie als een string.
    """
    info = ""
    part_of_ptm = False
    for line in lines:
        # Verwijder opmerkingen uit regels
        if "!" in line:
            line = line.split("!", 1)[0] + "\n"
        if f"RESI {ptm}" in line:
            part_of_ptm = True
            info = line
        elif "RESI" in line or "PRES" in line:
            part_of_ptm = False
        elif part_of_ptm:
            info += line
    return info

def obtain_separate_rtf(ptm, filepath):
    """
    Verkrijgt de RTF-informatie voor een specifieke PTM en schrijft deze naar aparte bestanden.

    :param ptm: Naam van de PTM.
    :param filepath: Pad naar het originele RTF-bestand.
    """
    lines = read_rtf_file(filepath)
    ptm_info = extract_ptm_info(ptm, lines)

    # Schrijf de geëxtraheerde PTM-informatie naar aparte bestanden
    write_to_file(f"{ptm}.txt", ptm_info)
    write_to_file(f"{ptm}_edited.txt", ptm_info)

def find_atom_connections(ptm):
    """
    Vindt de atom connecties in een PTM, identificerend enkele en dubbele bindingen.

    :param ptm: Naam van de PTM.
    :return: Lijst van tuples met atom en zijn connecties.
    """
    lines = read_rtf_file(f"{ptm}.txt")
    connections = []

    for line in lines:
        if "ATOM" in line and "ATOM H" not in line:
            single_bonds, double_bonds = [], []
            atom = line.split()[1]

            # Zoek naar bindingen en dubbele bindingen
            for bond_line in lines:
                if "BOND" in bond_line:
                    atoms = bond_line.split()[1:]
                    for x, y in zip(atoms[::2], atoms[1::2]):
                        if not (x.startswith("H") or y.startswith("H")):
                            if x == atom:
                                single_bonds.append(y)
                            elif y == atom:
                                single_bonds.append(x)
                elif "DOUBLE" in bond_line:
                    atoms = bond_line.split()[1:]
                    for x, y in zip(atoms[::2], atoms[1::2]):
                        if not (x.startswith("H") or y.startswith("H")):
                            if x == atom and y not in double_bonds:
                                double_bonds.append(y)
                            elif y == atom and x not in double_bonds:
                                double_bonds.append(x)
            connections.append((atom, single_bonds, double_bonds))
    print(f"Connections for {ptm}: {connections}")
    return connections

def update_atom_type(ptm, connections):
    """
    Werkt het atom type bij op basis van connecties.

    :param ptm: Naam van de PTM.
    :param connections: Lijst van tuples met atom en zijn connecties.
    """
    replacements = {
        # Voeg hier je atom type vervangingen toe
    }
    lines = read_rtf_file(f"{ptm}_edited.txt")
    updated_lines = []

    for line in lines:
        if "ATOM" in line:
            atom = line.split()[1]
            for connection in connections:
                if atom == connection[0]:
                    replacement = replacements.get(str(connection))
                    if replacement:
                        old_type = line.split()[2]
                        new_line = line.replace(old_type, replacement.ljust(len(old_type)))
                        updated_lines.append(new_line)
                    else:
                        updated_lines.append(line)
        else:
            updated_lines.append(line)

    print(f"Updated lines for {ptm}: {updated_lines[:5]}...")
    write_to_file(f"{ptm}_edited.txt", "".join(updated_lines))

def remove_hydrogen_from_bonds(ptm):
    """
    Verwijdert waterstofatomen uit bindingen in de PTM.

    :param ptm: Naam van de PTM.
    """
    lines = read_rtf_file(f"{ptm}_edited.txt")
    updated_lines = []

    for line in lines:
        if "BOND" in line:
            bonds = line.split()[1:]
            new_bonds = [bond for bond in bonds if not bond.startswith("H")]
            if new_bonds:
                updated_lines.append("BOND " + " ".join(new_bonds) + "\n")
        else:
            updated_lines.append(line)

    print(f"Lines after removing hydrogen bonds for {ptm}: {updated_lines[:5]}...")
    write_to_file(f"{ptm}_edited.txt", "".join(updated_lines))

def write_to_output_file(ptm, output_path):
    """
    Schrijft de bewerkte PTM-informatie naar een uitvoerbestand.

    :param ptm: Naam van de PTM.
    :param output_path: Pad naar het uitvoerbestand.
    """
    data = read_rtf_file(f"{ptm}_edited.txt")
    mode = 'a' if os.path.exists(output_path) else 'w'

    with open(output_path, mode) as file:
        file.write("\n\n" + "".join(data))
    print(f"Written to {output_path} for {ptm}")

def main():
    ptms = ["ARG"]
    rtf_filepath = "/Users/bashirking/PycharmProjects/pythonProject/ARG.txt"
    output_path = "/Users/bashirking/Desktop/Outputfile/practice_output_file.txt"

    for ptm in ptms:
        obtain_separate_rtf(ptm, rtf_filepath)
        connections = find_atom_connections(ptm)
        update_atom_type(ptm, connections)
        remove_hydrogen_from_bonds(ptm)
        write_to_output_file(ptm, output_path)

if __name__ == "__main__":
    main()
