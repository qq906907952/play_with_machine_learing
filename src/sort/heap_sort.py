def __heap_ele_sink(array, index,end):
    if 2 * index + 2 <= end :
        left_child = 2 * index + 1
        right_child = 2 * index + 2

        if array[left_child] > array[right_child] and array[left_child] > array[index]:
            array[index], array[left_child] = array[left_child], array[index]
            __heap_ele_sink(array, left_child,end)
        elif array[left_child] <= array[right_child] and array[right_child] > array[index]:
            array[index], array[right_child] = array[right_child], array[index]
            __heap_ele_sink(array, right_child,end)
    elif  2 * index + 1 <= end :
        if array[2 * index + 1]>array[index]:
            array[index],array[2*index+1]=array[2*index+1],array[index]
    else:
        return

def __init_heap(array):
    for i in range(((len(array)-1)-1)//2,-1,-1):
        __heap_ele_sink(array,i,len(array)-1)

def heap_sort(array):
    __init_heap(array)

    end = len(array) - 1
    while end>0:
        __heap_ele_sink(array,0,end)
        array[end],array[0]=array[0],array[end]
        end-=1


if __name__ == "__main__":
    import random
    import datetime

    a = [random.randint(-999999, 999999) for i in range(99999)]
    print("origin array:", a)
    now = datetime.datetime.now()
    heap_sort(a)
    print("time elapse:", datetime.datetime.now() - now)
    print("sorted array by {}".format(heap_sort.__name__))
    print("sorted array:", a)