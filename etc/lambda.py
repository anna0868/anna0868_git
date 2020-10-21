# lambda는 쓰고 버리는 일시적인 함수 입니다. 함수가 생성된 곳에서만 필요
# 간단한 기능을 일반적인 함수와 같이 정의해두고 쓰는 것이 아니고 필요한 곳에서 즉시 사용하고 버릴 수 있음


g = lambda x: x**2
print("g(8) : ", g(8))

f = lambda x, y: x + y
print("f(4, 4) : ", f(4, 4))

# 람다 함수 사용법
def inc(n):
	return lambda x: x + n

f = inc(2)
g = inc(4)
print("f(12) : ", f(12))
print("g(12) : ", g(12))

print("inc(2)(12) : ", inc(2)(12))




# map 함수와 함께 사용
'''map() 은 두 개의 인수를 가지는 함
r = map(function, iterable, ...)
첫 번째 인자 function 는 함수의 이름
두 번째 인자 iterable은 한번에 하나의 멤버를 반환할 수 있는 객체
(list, str, tuple) map() 함수는 function을 iterable의 모든 요소에 대해 적용 후 function에 의해 변경된 iterator를 반환'''



a = [1,2,3,4]
b = [17,12,11,10]
print("list(map(lambda x, y:x+y, a,b)) : ", list(map(lambda x, y:x+y, a,b)))

