import re
def is_named_entity(word):
    # Define patterns for entities (names, locations, organizations)
    name_pattern = re.compile(r'\b(?:శ్రీ|శ్రీమతి)?\.?\s*([\u0C01-\u0C7F]+)\b')

    # Modify location_pattern with more specific rules
    location_pattern = re.compile(r'\b(?:లో|నుంచి|దించి|వలన|అంగడి|గుడి)?\s*([\u0C01-\u0C7F]+)\b')

    organization_pattern = re.compile(r'\b(?:సంస్థ|కంపెనీ)?\s*([\u0C01-\u0C7F]+)\b')


    # Matching patterns in the word
    name_match = re.match(name_pattern, word)
    location_match = re.match(location_pattern, word)
    organization_match = re.match(organization_pattern, word)

    # Return True if any of the patterns match
    return name_match or location_match or organization_match

