#El problema de la mochila consiste en una lista de objetos en este caso, que tienen un valor y un peso determinados.
#El objetivo es llenar una mochila con un peso máximo de 15 kilos que sume el mayor valor posible
#Para ello voy  a hacer uso de la recursividad 


peso_maximo = 15
objects = [{"weight": 1, "value": 1},
           {"weight": 5, "value": 6},
           {"weight": 1, "value": 3},
           {"weight": 6, "value": 8},
           {"weight": 3, "value": 6},
           {"weight": 10, "value": 11},
           {"weight": 6, "value": 4},
           {"weight": 4, "value": 7},
           {"weight": 4, "value": 4},
           {"weight": 7, "value": 3}]
#Inicio la mejor mochila, el valor maximo y las iteraciones
mejor_mochila = []
valor_maximo = -1
n_iteraciones = 0


def llena_mochila(objetos,peso,mochila,valor):
    #Agrego las variables globales a la funcion 
    global mejor_mochila , valor_maximo,n_iteraciones

    n_iteraciones += 1

    #Genero una condición para cortar la iteración en caso de que el peso sea mayor del que puedo cargar

    if peso > peso_maximo:
        return
    #Si ya me he quedado sin objetos en la mochila compruebo el valor de la mochila actual y corto las iteraciones
    if not objetos:
        if valor > valor_maximo:
            mejor_mochila = mochila.copy()
            valor_maximo = valor
        return
    
    # En el primer caso NO agrego el objeto a la mochila
    llena_mochila(objetos[1:],peso,mochila,valor)
    objeto = objetos[0]
    nuevo_peso = peso + objeto["weight"]

    #Si no me paso de peso al agregar el objeto, cambio los valores y llamo a la funcion
    if nuevo_peso <= peso_maximo:
        c_mochila = mochila.copy()
        c_mochila.append(objeto)
        nuevo_valor = valor + objeto["value"]
        llena_mochila(objetos[1:],nuevo_peso,c_mochila,nuevo_valor)

llena_mochila(objects,0,[],0)
print(f"La mejor mochila ha sido: {mejor_mochila}")
print(f"Con un valor de: {valor_maximo}")
print(f"Con un peso de : {peso_maximo}")
print(f"Se ha completando en {n_iteraciones} iteraciones")
