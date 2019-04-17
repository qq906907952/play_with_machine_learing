def insert_sort(array):
    for i in range(1, len(array)):
        if array[i] < array[i - 1]:

            array[i], array[i - 1] = array[i - 1], array[i]
            i -= 1
            while i > 0 and array[i] < array[i - 1]:
                array[i], array[i - 1] = array[i - 1], array[i]
                i -= 1


if __name__ == "__main__":
    import random
    import datetime

    a=[random.randint(-999999,999999) for i in range(50000)]
    print("origin array:", a)
    now = datetime.datetime.now()
    insert_sort(a)
    print("time elapse:", datetime.datetime.now() - now)
    print("sorted array by {}".format(insert_sort.__name__))
    print("sorted array:",a)

