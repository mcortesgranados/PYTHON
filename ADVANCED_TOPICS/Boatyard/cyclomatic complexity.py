def calculate_grade(score):
    if score >= 90:
        grade = 'A'
    elif score >= 80:
        grade = 'B'
    elif score >= 70:
        grade = 'C'
    elif score >= 60:
        grade = 'D'
    else:
        grade = 'F'

    if grade == 'A':
        result = 'Excellent'
    elif grade == 'B':
        result = 'Good'
    elif grade == 'C':
        result = 'Average'
    elif grade == 'D':
        result = 'Below Average'
    else:
        result = 'Poor'

    return result
