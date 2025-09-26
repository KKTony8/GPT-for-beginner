def make_p(size,*toppings):
    print('\nSize of pizza: ' + str(size))
    print('Toppings: ')
    for topping in toppings:
        print('-' + topping)