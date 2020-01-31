def available_spots(lst, num):
    count = len(lst) - 1
    for i in range(len(lst) - 1):
        if num % 2 == 0:
            if lst[i] % 2 != 0 and lst[i + 1] % 2 != 0:
                count -= 1
        else:
            if lst[i] % 2 == 0 and lst[i + 1] % 2 == 0:
                count -= 1
    return count


def valid_name(name):
    if name.title() != name:  # 모든 첫 문자가 capitalized
        return False

    temp = name.split()

    if len(temp) not in [2, 3]:  # 이름은 적어도 2개 , 3개 여야
        return False

    if len(temp[-1]) < 3:  # 성은 words여야
        return False

    if len(temp[0]) and len(temp[1]) < 3:
        if temp[0] and temp[1].endswith('.'):
            pass
        else:
            return False

    return True