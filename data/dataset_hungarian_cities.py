# The original dataset is from https://simplemaps.com/data/world-cities
# It is under the Creative Commons Attribution 4.0 license.
ascii_lower_set = set(list("aábcdeéfghiíjklmnoóöőpqrstuvwxyuzs"))

cities = set()
with open('data/worldcities.csv', 'r', encoding='utf-8') as f:
    for line in f.readlines()[1:]:
        line = line.split(',')
        country = line[4].strip('"')
        if country == 'Hungary':
            city_raw = line[1].strip('"') # The city_ascii col, with the quotes removed
            # I only want cities that do not contain spaces, and only conatin latin letters,
            # with only lower case letters.
            city = city_raw.lower()
            print(city)
            if set(city).issubset(ascii_lower_set):
                cities.add(city)

if '' in cities:
    cities.remove('')

with open('data/cities_normalized_hungary.txt', 'w') as f:
    f.write('\n'.join(cities))