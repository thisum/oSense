from tkinter import *
from test import print_line


root = Tk()

frame = Frame(root)
frame.pack()

pred_var = StringVar()
posture_var = StringVar()
activity_var = StringVar()

bottomframe = Frame(root)
bottomframe.pack( side = BOTTOM )


lblFinalPrediction = Label(frame, textvariable=pred_var, justify="center")
lblFinalPrediction.config(font=("Courier", 500))
lblFinalPrediction.pack( fill=X,padx=50 )

lblPosture = Label(bottomframe, text="Posture Predictions: ", fg="red")
lblPosture.pack(side = LEFT)

lblPostureResult = Label(bottomframe, textvariable=posture_var, fg="blue")
lblPostureResult.pack( side = LEFT )

lblAction = Label(bottomframe, text="Action Predictions :", fg="brown")
lblAction.pack( side = LEFT )

lblActionResult = Label(bottomframe, textvariable=activity_var, fg="blue")
lblActionResult.pack( side = LEFT )

# thread1 = classifierThread()
# thread1.start()

print_line()

root.mainloop()