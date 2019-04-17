def __quick_sort(array, start, end):
    if end - start <= 0:
        return
    elif end - start == 1:
        if array[end] < array[start]:
            array[start], array[end] = array[end], array[start]
    else:
        first = array[start]
        i, j = start, end

        while i < j:
            while i < j and array[j] > first:
                j -= 1

            while i < j and array[i] <= first:
                i += 1

            array[i], array[j] = array[j], array[i]

        array[start], array[i] = array[i], array[start]

        __quick_sort(array, start, i - 1)
        __quick_sort(array, i + 1, end)

def quick_sort(array):
    __quick_sort(array,0,len(array)-1)



if __name__ == "__main__":
    import random

    a=[random.randint(-9999,9999) for i in range(99999)]
    print("origin array:",a)
    quick_sort(a)
    print("sorted array:",a)