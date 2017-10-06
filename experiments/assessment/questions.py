"""
This file holds the two questions asked to 
each subject during the visual assessment
"""
import sys
if sys.version_info > (3, 0):
    from tkinter import *
else:
    from Tkinter import *  # paleolithic python version here


def center(toplevel):
    """
    Centers the tkinter window. 
    Copy pasted from [1] with no further reasoning.
    """

    toplevel.update_idletasks()
    w = toplevel.winfo_screenwidth()
    h = toplevel.winfo_screenheight()
    size = tuple(int(_) for _ in toplevel.geometry().split('+')[0].split('x'))
    x = w/2 - size[0]/2
    y = h/2 - size[1]/2
    toplevel.geometry("%dx%d+%d+%d" % (size + (x, y)))


def ask_question(question, answers):
    """
    Asks a generic question.
    
    Parameters
    ----------
    question: str
        the text of the question.
    answers: dict
        the possible answers in the form of a dictionary like {value: text_in_radio_button}.

    Returns
    -------
    int
        the key of the choice in the answers dictionary.
    """
    root = Tk(baseName='QUESTION 1')

    var = IntVar(value=0)
    frm = Frame(root, bd=16)

    T = Text(frm)
    T.insert(END, question)
    T.config(state=DISABLED)
    T.pack()

    frm.pack()
    for answer in answers:
        r = Radiobutton(frm, text=answers[answer], bd=4, width=12)
        r.config(indicatoron=0, variable=var, value=answer)
        r.pack(side='left')

    def submit():
        root.quit()
        root.destroy()

    Button(text='Submit', command=submit).pack(fill=X)
    center(root)
    root.mainloop()

    return var.get()


def ask_question_1():
    """
    Asks the first qualitative assessment question.
    """
    question = 'If you were sitting in the same car whose driver ' \
                    'exhibits the attentional behavior you\'ve just seen), \n How safe would you feel? [1-5]'
    answers = {i: str(i) for i in range(1, 5 + 1)}

    return ask_question(question=question, answers=answers)


def ask_question_2():
    """
    Asks the second qualitative assessment question.
    """
    question = 'Do you think you observed the attentional behavior of a human or a AI?'
    answers = {0: 'Human', 1: 'AI'}

    return answers[ask_question(question=question, answers=answers)]


# simple test case
if __name__ == '__main__':
    print(ask_question_1())
    print(ask_question_2())


"""
References
----------
[1] 'https://stackoverflow.com/questions/3352918/how-to-center-a-window-on-the-screen-in-tkinter'
"""