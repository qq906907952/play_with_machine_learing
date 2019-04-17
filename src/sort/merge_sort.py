def __merge(array, start, middle, end):
    temp = []
    i, j = start, middle + 1
    while i <= middle and j <= end:
        if array[i] < array[j]:

            temp.append(array[i])
            i += 1
        else:
            temp.append(array[j])
            j += 1

    while i <= middle:
        temp.append(array[i])
        i += 1
    while j <= end:
        temp.append(array[j])
        j += 1

    array[start:end + 1] = temp

def __merge_sort(array, start, end):
    if end - start <= 0:
        return
    elif end - start == 1:

        if array[end] < array[start]:
            array[start], array[end] = array[end], array[start]

    else:
        __merge_sort(array, start, (start + end) // 2)
        __merge_sort(array, (start + end) // 2 + 1, end)

        __merge(array, start, (start + end) // 2, end)

def merge_sort(array):
    __merge_sort(array,0,len(array)-1)

if __name__ == "__main__":
    import random
    import datetime

    a = [random.randint(-999999, 999999) for i in range(99999)]
    print("origin array:", a)
    now = datetime.datetime.now()
    merge_sort(a)
    print("time elapse:", datetime.datetime.now() - now)
    print("sorted array by {}".format(merge_sort.__name__))
    print("sorted array:", a)
