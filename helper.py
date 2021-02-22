from datetime import date 

def calculateAge(dob): 
    # date(year,month,day)    
    dob = dob.split("-")
    # print(dob)
    y = int(dob[0])
    m = int(dob[1])
    d = int(dob[2])    
    birthDate = date(y, m, d)
    days_in_year = 365.2425    
    age = int((date.today() - birthDate).days / days_in_year) 
    return age 


