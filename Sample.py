# values = {float(9.0),int(9)} # set values as {("float",9.0),("int",9)} or {"9",9.0} or {9,"9.0"}
# print(values)

# n = int(input("Enter a number: "))
# i=1
# sum=0
# while i<=n:
#     sum=sum+i
#     i+=1
# print(sum)

n = int(input("Enter a number: "))
m=1
for i in range(1,n+1):
    m = m*i

print(m)