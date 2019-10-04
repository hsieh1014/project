import tkinter as tk
#window
window = tk.Tk()
window.title('title')
window.geometry('500x500')
window.configure(background="#FFFFFF")
#label
a = 100
var = tk.StringVar()        
l1 = tk.Label(window,text=a,bg='#8AA8A1',fg='white',
              font=('Arial',12),width=30,height=2).place(x=155, y=350, anchor='nw')
l2 = tk.Label(window,textvariable=var,bg='#4A5859',fg='white',
              font=('Arial',12),width=30,height=2).place(x=155, y=390, anchor='nw')

#button function
hit = False
def add(i):
    global hit
    var.set(i)
    if hit == False:
        hit = True
        i = i+1
        var.set('i')
        return i
    else:
        hit = False
        var.set('error')
        
#button
b = tk.Button(window,text="+",font=('Arial',12),
              width=10,height=1,command=add(a)).place(x=225, y=450, anchor='nw')

#window
window.mainloop()
