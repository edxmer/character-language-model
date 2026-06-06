# The original dataset is from https://simplemaps.com/data/world-cities
# It is under the Creative Commons Attribution 4.0 license.
import string
ascii_lower_set = set(string.ascii_lowercase + ' ')

cities = set()
with open('src/data/original/worldcities.csv', 'r', encoding='utf-8') as f:
    for line in f.readlines()[1:]:
        line = line.split(',')
        country = line[4].strip('"')
        if country in ['United States', 'United Kingdom', 'Australia']:
            city_raw = line[1].strip('"') # The city_ascii col, with the quotes removed
            # I only want cities that do not contain spaces, and only conatin latin letters,
            # with only lower case letters.
            city = city_raw.lower()
            #print(city)
            if set(city).issubset(ascii_lower_set):
                cities.add(city)

cities.discard('')

with open('src/data/normalized/en_cities.txt', 'w') as f:
    f.write('\n'.join(cities))