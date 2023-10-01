N,M,B=input().split()
block=[]

for i in range(int(N)):
    block.append(list(map(int, input().split())))


def def_1(a,b,Time,B):
    
    Time+=(a-b)*2
    B+=a-b
def def_2(a,b,Time,B,N,M,temp):
    if B>0:
        Time+=(b-a)
        B-=b-a
    else:
        temp.append([N,M])
max_block=0
min_block=255
for x in range(len(block)):
    temp_max=max(block[x])
    temp_min=min(block[x])
    if max_block<temp_max:
        max_block=temp_max
        
    if temp_min<temp_min:
        min_block=temp_min
record=[]
for b in range(temp_min,max_block):
    Time=0
    temp=[]
    while(1):
        for x in range(N):
            for y in range(M):
                if block[x][y]>b:
                    def_1(block[x][y],b,Time)
                else:
                    def_2(block[x][y],b,Time,B,temp)
        for x,y in (temp):
            tmp=[]
            def_2(block[x][y], b, Time, B,tmp)
            if len(tmp)==0:
                record=[b,Time]
                if record[1]==Time:
                    record[0]=b
                
        break      
print(record[1],record[0])