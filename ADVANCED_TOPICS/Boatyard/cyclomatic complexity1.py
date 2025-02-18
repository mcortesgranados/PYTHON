def calculate_grade(score):
    grade_mapping = {
        90: ('A', 'Excellent'),
        80: ('B', 'Good'),
        70: ('C', 'Average'),
        60: ('D', 'Below Average')
    }

    for cutoff, (grade, result) in grade_mapping.items():
        if score >= cutoff:
            return grade, result

    return 'F', 'Poor'
