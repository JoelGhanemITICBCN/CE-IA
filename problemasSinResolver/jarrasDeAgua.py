required_water = 4
water_pitchers = [3, 5]
def water_pitcher(necesaria,jarras):
    a = 0
    b = 0
    jarras.sort()
    if necesaria > max(jarras):
        return "Es imposible"
    #while b != necesaria:
    b = jarras[1]
    print(f"a: {a}  b:{b}")
    a = jarras[0]
    b = b - a
    print(f"b: {b} b-a:{b-a}")
    a = 0
    a = b
    b = jarras[1]
    b = b - a
    return b
water_pitcher(required_water,water_pitchers)
