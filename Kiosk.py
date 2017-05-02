from Tkinter import Tk, Frame, BOTH, StringVar
import Tkinter
from PIL import Image, ImageTk
import os
import readchar
import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)
GPIO.setup(17,GPIO.IN,pull_up_down=GPIO.PUD_UP)



class Example(Frame):

        def __init__(self, parent):
		Frame.__init__(self, parent)            
		self.parent = parent        
		self.initUI()
		

	def initUI(self):
		self.parent.title("PISE")
		#self.parent.wm_overrideredirect(True)
		#self.pack(fill=BOTH, expand=1)


        
                
	
root = Tk()
root.geometry("700x400+-20+-20")
app = Example(root)

im = Image.open('/home/pi/Desktop/test.jpg')
tkimage = ImageTk.PhotoImage(im)
myvar=Tkinter.Label(root,image = tkimage)
myvar.place(x=0, y=0, relwidth=1, relheight=1)

#myvar2=tkinter.Label(text="hello world").grid(row=1, column=5)

info = StringVar()
info.set('Water Quality Analyzer')

myvar2=Tkinter.Label(textvariable=info,width=200,height=1,font=("Courier",30)).pack(side="left")

root.focus_set()

def yourFunction():

        print("One Moment Please...")
        GPIO.wait_for_edge(17,GPIO.RISING)
        info.set("One Moment Please...")
        os.system("sudo raspistill -o image.jpg")
        time.sleep(2)
        info.set("8 red and 1 blue")
        root.after(2000,yourFunction)
	
root.bind("<k>",yourFunction)

def functioning(event):
        i = 0
        print (i)
        i = i+1
        os.system("raspistill -o image.jpg --timeout 1")
        print(i)
        i = i+1
        print (i)
	info.set("8 red and 1 blue")


#GPIO.add_event_detect(17,GPIO.FALLING,callback=functioning)



#custName = StringVar(None)
#yourName = Entry(root, textvariable=custName)
#yourName.pack()

#relStatus = StringVar()
#relStatus.set(None)

#labelText = StringVar()
#labelText.set('Accuracy Level')
#label1 = Label(root, textvariable=labelText, height=2)
#label1.pack()

#def beenClicked1():
	#pass

#def beenClicked5():
	#pass

#radio1 = Radiobutton(root, text='100%', value='1', variable = relStatus, command=beenClicked1).pack()
#radio2 = Radiobutton(root, text='50%', value='5', variable = relStatus, command=beenClicked5).pack()

root.after(2000,yourFunction)
root.mainloop()
