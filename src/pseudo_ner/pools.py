"""Grands pools de valeurs pour la ré-injection dans les décisions réelles.

Pourquoi un pool séparé du corpus synthétique : une décision réelle contient en moyenne 45 mentions de
personnes. Avec une liste de 25 noms, chaque nom revient des centaines de fois dans l'entraînement et le
modèle apprend *la liste* au lieu du *contexte* (« Monsieur ___ », « née le ») — constaté : rappel 0,13 sur des
noms jamais vus. Avec plusieurs centaines de noms répartis en trois parts disjointes (train / dev / test),
chaque nom n'apparaît que dans une ou deux décisions et le contexte redevient le seul signal fiable.

Les listes ci-dessous sont des noms, prénoms, communes et voies **courants en France** ; aucune personne
réelle n'est visée. Les valeurs réservées au jeu de test synthétique (``HELD_OUT_*``) en sont exclues pour que
« jamais vu » garde son sens d'un corpus à l'autre.
"""

from __future__ import annotations

import hashlib

from pseudo_ner import synthetic as syn
from pseudo_ner.synthetic import Pools


def _list(text: str) -> list[str]:
    """Mots séparés par des blancs ; un ``_`` marque une espace interne (« Le_Mans » → « Le_Mans »)."""
    return [w.replace("_", " ") for w in text.split()]


NOMS = _list("""Martin Bernard Thomas Petit Robert Richard Durand Dubois Moreau Laurent Simon Michel Lefebvre Leroy Roux
David Bertrand Morel Fournier Girard Bonnet Dupont Lambert Fontaine Rousseau Vincent Muller Lefevre Faure André
Mercier Blanc Guérin Boyer Garnier Chevalier François Legrand Garcia Robin Nicolas Henry Roussel Masson Lemaire
Marie Lemaître Blanchard Giraud Rivière Brunet Gaillard Arnaud Martinez Gérard Renard Schmitt Leroux Colin Vidal
Caron Picard Fabre Aubert Lemoine Renaud Lacroix Olivier Philippe Bourgeois Pierre Benoit Rey Payet Rolland
Leclercq Guillaume Lecomte Lopez Dupuy Guillot Hubert Berger Carpentier Sanchez Dupuis Louis Deschamps Huet
Vasseur Boucher Fleury Royer Klein Jacquet Adam Paris Poirier Marty Aubry Guyot Carré Charles Renault
Charpentier Ménard Maillard Baron Bertin Bailly Hervé Schneider Fernandez Collet Léger Bouvier Julien Prévost
Millet Perrot Daniel Cousin Germain Breton Besson Langlois Rémy Pelletier Lévêque Perrier Leblanc Barré Lebrun
Weber Mallet Hamon Marques Gillet Bousquet Delmas Mahé Monnier Bazin Collin Étienne Grondin Thibault Chauvin
Boulanger Tessier Hoarau Gomez Humbert Blin Bigot Laporte Delaunay Lainé Devaux Texier Bénard Rodriguez Seguin
Chevallier Pons Morvan Lacombe Valentin Lagarde Pascal Pichon Cordier Gros Ferreira Cohen Besnard Michaud Jacob
Lejeune Gonzalez Lamy Camus Marin Guichard Pasquier Maillot Barbe Ollivier Antoine Bonneau Leduc Sauvage
Chartier Lévy Benoist Lebreton Blanchet Leconte Ruiz Lacoste Hernandez Martins Prieur Fischer Poulain Wagner
Gilbert Lenoir Marchetti Delorme Bouchet Rousset Hardy Buisson Guillou Alves Raynaud Pereira Jourdan Coulon
Vallet Peltier Charrier Boutin Rossi Delahaye Lebon Couturier Descamps Maurice Lebas Dupré Bruneau Blot Gay
Jacquot Godard Tanguy Carlier Mangin Laroche Didier Costa Bouchard Ribeiro Allard Dubosc Verdier Vial Serre
Bodin Flament Grenier Chauveau Jouve Moreno Hamel Lecoq Guyard Barthélemy Roland Normand Gosselin Delattre
Thierry Clerc Pruvost Gallet Lescure Bonnin Teixeira Boulay Bernier Pasquet Jaouen Riou Le_Bihan Le_Guen
Bellec Guégan Le_Floch Kerbrat Quéré Salaün Abgrall Corre Le_Meur Le_Berre Jaffré Cariou Kerouanton Goasdoué
Mazé Brault Guilbert Dreyfus Wolff Hirsch Kahn Zimmermann Keller Hoffmann Kuhn Braun Meier Bauer Huber
Sommer Frey Roth Walter Stein Lang Krieger Haas Baumann Ferrand Galland Chaumont Raymond Maury Joubert
Fabien Delacroix Lamarque Castel Albert Portal Vernet Puech Cazals Delpech Rouquette Bousquié Soulié Laffont
Combes Fraisse Azéma Sirven Bonnafé Nadal Amiel Ricard Gasc Aussel Delbos Carrère Dupouy Lassalle Darrieu
Lacaze Bordes Castaing Labat Barrière Mathis Riera Ramos Nunes Carvalho Oliveira Mendes Santos Pinto Correia
Moura Fonseca Benali Haddad Belkacem Saidi Bouaziz Ziani Rahmani Benamar Brahimi Hamidi Kaci Mansouri Touati
Amrani Cherif Ouali Bensaid Nguyen Tran Pham Diallo Traoré Keita Camara Sow Ndiaye Diop Cissé Konaté Sylla
Coulibaly Dembélé Fofana Yilmaz Demir Kaya Celik Ozturk Aydin Sahin""")

PRENOMS_F = _list("""Marie Sophie Isabelle Nathalie Camille Catherine Céline Valérie Fatima Nadia Sandrine Aurélie Élodie
Stéphanie Émilie Laetitia Hélène Anaïs Inès Léa Chloé Leïla Amina Louise Martine Sylvie Christine Véronique
Monique Françoise Jacqueline Nicole Dominique Chantal Annie Patricia Corinne Laurence Karine Virginie Audrey
Julie Mélanie Marion Pauline Charlotte Mathilde Margaux Clara Emma Jade Lina Sarah Lucie Alice Juliette Zoé
Eva Lola Rose Anna Lisa Nina Maëlys Romane Élise Jeanne Agathe Victoire Capucine Noémie Laura Marina
Coralie Fanny Maud Ophélie Morgane Gwenaëlle Solène Anne Sabine Béatrice Carole Nadine Muriel Josiane
Michèle Odile Danielle Colette Simone Thérèse Madeleine Suzanne Yvonne Germaine Lucienne Paulette Huguette
Denise Ginette Raymonde Khadija Samira Souad Nawel Myriam Rachida Zineb Sabrina Assia Hanane Malika Houda
Aïcha Fatoumata Aminata Awa Mariam Kadiatou Oumou Thi Linh Mai Ayse Elif Zeynep Ana Maria Carmen Lucia
Rosa Paula Joana Teresa Sandra Vanessa Jessica Cindy Priscilla Mélissa Marjorie Séverine Ingrid Estelle
Hortense Clémence Apolline Garance Bérénice Eugénie Joséphine Blanche Violette Iris Léonie Salomé Maya
Alix Diane Margot Ambre Louna Mila Léna Lou Candice Célia Elsa Erika Flora Gaëlle Ilona Jeannine Karima
Kenza Lila Lydie Maëva Nelly Oriane Perrine Rachel Roxane Sonia Tiphaine Valentine Wendy Yolande""")

PRENOMS_M = _list("""Jean Pierre Nicolas Philippe Thomas Julien Laurent Antoine Mohamed Karim Alexandre Sébastien
Christophe Olivier Patrick Michel Alain Bernard Daniel Jacques Gérard André Marcel René Georges Louis
Henri Robert Paul Maurice Raymond Roger Lucien Marc Éric Pascal Frédéric Bruno Thierry Didier Franck Vincent
Guillaume Mathieu Maxime Romain Benjamin Kévin Jérémy Anthony Jonathan Quentin Hugo Lucas Nathan Enzo Léo
Gabriel Raphaël Arthur Louis Jules Adam Noah Ethan Tom Théo Clément Baptiste Simon Martin Victor Samuel Paul
Augustin Côme Gaspard Oscar Marius Félix Léon Émile Eugène Armand Fernand Gustave Alfred Albert Édouard
Charles Joseph François Christian Dominique Yves Serge Gilles Hervé Denis Joël Yannick Loïc Erwan Gwendal
Ronan Yoann Tanguy Corentin Malo Ewen Brieuc Rachid Youssef Omar Bilal Samir Nabil Hamza Amine Yassine
Ismaël Ibrahim Abdel Sofiane Ilyes Walid Farid Hakim Idriss Moussa Mamadou Ousmane Abdoulaye Boubacar Sékou
Amadou Ali Hassan Hussein Mustapha Ahmed Khalil Tarek Nassim Anis Mehdi Van Minh Duc Mehmet Mustafa Emre
Murat Ali Carlos José Miguel Manuel Luis Pedro Antonio Jorge Rui Tiago Diogo Francisco Fabrice Ludovic
Cyril Fabien Grégory Mickaël Steve Jordan Dylan Kylian Noé Timéo Sacha Mathis Evan Nolan Axel Robin Eliott
Rayan Aaron Ayoub Elias Isaac Tristan Thibaut Guilhem Aurélien Gaëtan Geoffrey Valentin Alexis Florent""")

VILLES = _list("""Paris Marseille Lyon Toulouse Nice Nantes Montpellier Strasbourg Bordeaux Lille Rennes Reims Toulon
Grenoble Dijon Angers Nîmes Clermont-Ferrand Le_Mans Aix-en-Provence Brest Tours Amiens Limoges Annecy
Perpignan Boulogne-Billancourt Metz Besançon Orléans Rouen Argenteuil Montreuil Caen Nancy Saint-Denis
Roubaix Tourcoing Nanterre Vitry-sur-Seine Créteil Avignon Poitiers Aubervilliers Dunkerque Versailles
Aulnay-sous-Bois Courbevoie Colombes Asnières-sur-Seine Cherbourg Saint-Pierre Rueil-Malmaison Pau
Champigny-sur-Marne Antibes Calais Béziers La_Seyne-sur-Mer Saint-Nazaire Mérignac Drancy Ajaccio Cannes
Noisy-le-Grand Vénissieux Cergy Pessac Levallois-Perret Ivry-sur-Seine Clichy Issy-les-Moulineaux Évry
Villejuif Troyes Antony Neuilly-sur-Seine Sarcelles Niort Pantin Chambéry Le_Blanc-Mesnil Lorient Beauvais
Hyères Meaux Montauban Chelles Narbonne Villepinte Épinay-sur-Seine Bobigny Cholet Saint-Quentin Fréjus
Bondy Sartrouville Arles Vannes Clamart Sevran Mulhouse Blois Bayonne Évreux Charleville-Mézières Laval
Albi Belfort Brive-la-Gaillarde Carcassonne Agen Chartres Colmar Quimper Valence Bourges Angoulême
Châteauroux Compiègne Tarbes Mâcon Périgueux Auxerre Saint-Malo Cahors Rodez Mende Guéret Aurillac Tulle
Digne-les-Bains Gap Foix Vesoul Épinal Bar-le-Duc Chaumont Nevers Moulins Montluçon Vichy Roanne Vienne
Bourg-en-Bresse Oyonnax Thonon-les-Bains Annemasse Albertville Montélimar Orange Carpentras Salon-de-Provence
Martigues Istres Aubagne La_Ciotat Draguignan Grasse Menton Bastia Corte Sète Agde Lunel Alès Lodève
Millau Castres Mazamet Auch Mont-de-Marsan Dax Libourne Arcachon Bergerac Sarlat Périgueux Cognac Saintes
Rochefort Royan Châtellerault Thouars Bressuire Fontenay-le-Comte Les_Sables-d'Olonne Challans Saumur Segré
Mayenne Château-Gontier Fougères Vitré Redon Dinan Lannion Morlaix Landerneau Concarneau Pontivy Auray
Alençon Flers Lisieux Vire Bayeux Honfleur Dieppe Fécamp Le_Havre Elbeuf Louviers Vernon Dreux Nogent-le-Rotrou
Vendôme Romorantin Montargis Pithiviers Sens Joigny Avallon Beaune Chalon-sur-Saône Le_Creusot Autun Dole
Lons-le-Saunier Pontarlier Montbéliard Lure Saint-Dié Remiremont Toul Lunéville Thionville Forbach Sarreguemines
Haguenau Saverne Sélestat Obernai Molsheim Laon Soissons Château-Thierry Senlis Creil Abbeville Péronne
Arras Lens Douai Cambrai Valenciennes Maubeuge Béthune Boulogne-sur-Mer Saint-Omer Hazebrouck""")


def _voies() -> list[str]:
    raw = """rue de la République;avenue de la Gare;boulevard Victor-Hugo;rue Jean-Jaurès;place de l'Église;
rue du Général-de-Gaulle;avenue Jean-Moulin;rue Pasteur;rue des Lilas;allée des Tilleuls;impasse des Rosiers;
chemin des Vignes;route de Lyon;rue Gambetta;avenue Aristide-Briand;rue du Moulin;rue de l'Industrie;
rue Émile-Zola;boulevard de Strasbourg;rue des Acacias;avenue du Maréchal-Leclerc;rue Lamartine;rue Jules-Ferry;
rue Anatole-France;rue de la Paix;rue du Commerce;rue des Jardins;place de la Mairie;rue Nationale;rue Carnot;
quai de la Loire;cours Lafayette;rue de Verdun;avenue de la Liberté;rue Diderot;rue Molière;rue Racine;
rue Danton;rue Marceau;rue Kléber;rue Hoche;rue Lafayette;rue Saint-Jacques;rue Saint-Pierre;rue Sainte-Anne;
rue du Château;rue de l'Hôtel-de-Ville;rue du Port;rue de la Mer;avenue des Pins;allée des Chênes;
chemin du Bois;rue des Peupliers;rue des Platanes;rue des Érables;rue des Cerisiers;rue des Pommiers;
rue des Mimosas;rue des Glycines;rue des Bleuets;rue des Coquelicots;square des Violettes;clos des Lauriers;
résidence les Hauts;lotissement le Clos;avenue de Paris;route de Bordeaux;route de Nantes;avenue de Toulouse;
rue de Lille;rue de Metz;rue de Rennes;rue de Rouen;rue d'Alsace;rue de Bretagne;rue de Provence;
rue de Lorraine;rue de Picardie;rue de Normandie;rue de Savoie;rue du 8-Mai-1945;rue du 11-Novembre;
rue du 14-Juillet;avenue du 19-Mars-1962;rue Pierre-Curie;rue Marie-Curie;rue Louis-Pasteur;rue Jean-Monnet;
rue Robert-Schuman;rue Léon-Blum;rue Jean-Mermoz;rue Saint-Exupéry;rue Guy-Môquet;rue Pierre-Brossolette;
rue Jean-Baptiste-Clément;rue Paul-Bert;rue Paul-Langevin;rue Henri-Barbusse;rue Romain-Rolland;
rue Pierre-Sémard;rue Marcel-Cachin;rue Maurice-Thorez;boulevard de la Liberté;boulevard Gambetta;
boulevard Haussmann;boulevard Saint-Germain;avenue des Champs-Élysées;avenue Victor-Hugo;rue des Fleurs;
rue des Écoliers;rue de la Fontaine;rue du Lavoir;rue de la Forge;rue du Stade;rue de la Gare;rue du Pont"""
    return [v.strip() for v in raw.replace("\n", "").split(";") if v.strip()]


def _dedupe(values: list[str], exclude: list[str]) -> list[str]:
    seen = {e.casefold() for e in exclude}
    out = []
    for v in values:
        key = v.casefold()
        if key not in seen:
            seen.add(key)
            out.append(v)
    return out


def _bucket(value: str) -> int:
    return int(hashlib.sha256(value.casefold().encode()).hexdigest(), 16) % 5


def _part(values: list[str], part: str, exclude: list[str]) -> list[str]:
    """Partage déterministe par empreinte : ``dev`` = 1/5, ``test`` = 1/5, ``train`` = 3/5 — après retrait de ``exclude``."""
    wanted = {"dev": {0}, "test": {1}, "train": {2, 3, 4}}[part]
    return [v for v in _dedupe(values, exclude) if _bucket(v) in wanted]


def pool(part: str) -> Pools:
    """Trois pools disjoints. ``test`` ajoute les valeurs ``HELD_OUT_*`` du synthétique et exclut tout ce que le
    synthétique d'entraînement a vu, pour que « jamais vu » reste vrai d'un corpus à l'autre."""
    if part not in ("train", "dev", "test"):
        raise ValueError(f"pool inconnu : {part}")
    held = Pools(
        syn.HELD_OUT_PRENOMS_F, syn.HELD_OUT_PRENOMS_M, syn.HELD_OUT_NOMS, syn.HELD_OUT_VILLES, syn.HELD_OUT_VOIES, syn.HELD_OUT_SOCIETES
    )
    seen_by_synthetic = Pools(syn.PRENOMS_F, syn.PRENOMS_M, syn.NOMS, syn.VILLES, syn.VOIES, syn.SOCIETES_NOM)
    big = Pools(PRENOMS_F, PRENOMS_M, NOMS, VILLES, _voies(), syn.SOCIETES_NOM)
    out = {}
    for field in ("prenoms_f", "prenoms_m", "noms", "villes", "voies", "societes"):
        exclude = getattr(held, field) + (getattr(seen_by_synthetic, field) if part == "test" else [])
        values = _part(getattr(big, field), part, exclude)
        out[field] = getattr(held, field) + values if part == "test" else values
    return Pools(**out)
