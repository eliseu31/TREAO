
fact_dict = {}


def factorial(n):
    # use recursive function to test the memory usage
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)


def factorial_memo(n):
    if n == 0:
        return 1
    else:
        if n-1 not in fact_dict:
            fact_dict[n - 1] = factorial(n-1)

        return n * fact_dict[n - 1]


for i in range(100000):
    factorial(800)
