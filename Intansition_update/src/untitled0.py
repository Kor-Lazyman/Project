class Equipment:
    def __init__(self,Name,ID):
        self.Name=Name
        self.ID=ID
        self.Power='Off'
    def Power_On(self):
        self.Power='On'
    def Power_Off(self):
        self.Power='Off'

Notice="=====Notice=====\nCmd 0: List of Equipment\nCmd 1:Equipment On\nCmd 2: Equipment Off\nCmd 3:All Equipment On\nCmd 4 All Equipment off\nCmd 5:Program Shutdown" 
Class_list=[]
Shutdown=0
Num_of_Equipment=int(input())
for Count in range(Num_of_Equipment):
    Name=input()
    Class_list.append(Equipment(Name,Count))
    
while(Shutdown==0):
    print(Notice)
    Cmd=int(input())
    if Cmd==0:
        for Equip in Class_list:
            print("ID:",Equip.ID,"Name:",Equip.Name,"\tPower",Equip.Power)
    elif Cmd==1:
        while(True):
            Equip_num=int(input())
            if Equip_num<Num_of_Equipment and Equip_num>0:
                Class_list[Equip_num].Power_On()
                break
            else:
                print("0이상 ",Num_of_Equipment,"미만의 숫자로 입력해주십시오")
    elif Cmd==2:
        while(True):
            Equip_num=int(input())
            if Equip_num<Num_of_Equipment and Equip_num>0:
                Class_list[Equip_num].Power_Off()
                break
            else:
                print("0이상 ",Num_of_Equipment,"미만의 숫자로 입력해주십시오")
    elif Cmd==3:
        for Equip in Class_list:
            Equip.Power_On()
    
    elif Cmd==4:
        for Equip in Class_list:
            Equip.Power_Off()
    
    elif Cmd==5:
        Shutdown=1
    else:
        print("0이상 ",Num_of_Equipment,"미만의 숫자로 입력해주십시오")
    
print("프로그램 종료")