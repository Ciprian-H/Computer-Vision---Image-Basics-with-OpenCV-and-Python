#!/usr/bin/env python
# coding: utf-8

# ### Task 1: How to open an Image with Matplotlib
# ### Task 2: Get familiar with RGB channels
# ### Task 3: Differences between Matplotlib and OpenCV
# ### Task 4: Resize & Flip an Image
# ### Task 5: Draw Shapes on an Image
# ### Task 6: Draw with the Mouse
# ### Task 7: Event Choices for the Mouse
# ### Task 8: Mouse Functionality

# ## 1. Matlpotlib & PIL

# #### Import Libraries

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# #### Use the PIL for the image

# In[2]:


from PIL import Image


# In[3]:


img = Image.open('rock.jpg')


# In[4]:


img


# #### Rotate the image

# In[5]:


img.rotate(-90)


# #### Check the type of the image

# In[6]:


type(img)


# #### Turn the image into an array

# In[7]:


img_array = np.asarray(img)


# In[8]:


type(img_array)


# #### Get the Height, Width & Channels

# In[9]:


img_array.shape


# In[10]:


plt.imshow(img_array)


# ## Task 2: Get familiar with RGB channels

# #### R G B channels
# 
# ##### Red channel is in postion No. 0
# ##### Green channel is in postion No. 1
# ##### Blue channel is in postion No. 2
# 
# The Colour are from 0 == no color from the channel, to 255 == full color from channel

# In[11]:


img_test=img_array.copy()


# #### Only Red channel

# In[12]:


plt.imshow(img_test[:,:,0])


# In[13]:


# Scale Red channel to Gray
plt.imshow(img_test[:,:,0], cmap='gray')


# #### Only Green Red channel

# In[14]:


plt.imshow(img_test[:,:,1])


# In[15]:


# Scale Green channel to Gray
plt.imshow(img_test[:,:,1], cmap='gray')


# #### Only Blue Red channel

# In[16]:


plt.imshow(img_test[:,:,2])


# In[17]:


# Scale Blue channel to Gray
plt.imshow(img_test[:,:,2], cmap='gray')


# #### Remove the Red color

# In[18]:


img_test[:,:,0]=0
plt.imshow(img_test)


# #### Remove the Green color

# In[19]:


img_test[:,:,1]=0
plt.imshow(img_test)


# #### Remove the Blue color

# In[20]:


img_test[:,:,2]=0
plt.imshow(img_test)


# ## Task 3: Differences between Matplotlib and OpenCV

# ### Import OpenCV

# In[21]:


import cv2


# #### Get the image with the imread

# In[22]:


img = cv2.imread(r'C:\Users\hancu\OneDrive - Nokia\Trainings TBD\Computer Vision - Image Basics with OpenCV and Python\rock.jpg')


# #### Image type

# In[23]:


type(img)


# #### Image shape

# In[24]:


img.shape


# In[25]:


plt.imshow(img)


# #### Until now we were working with Matplotlib and RGB
# #### OpenCV is reading the channel as BGR
# #### We will convert OpenCV to the channel of the photo.

# In[26]:


img_fix = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# In[27]:


plt.imshow(img_fix)


# #### Scale it to Gray and check the shape

# In[28]:


img_gray = cv2.imread(r'C:\Users\hancu\OneDrive - Nokia\Trainings TBD\Computer Vision - Image Basics with OpenCV and Python\rock.jpg',
                     cv2.IMREAD_GRAYSCALE)


# In[29]:


img_gray.shape


# In[30]:


plt.imshow(img_gray, cmap='gray')


# #### Resize the image

# In[31]:


img_new = cv2.resize(img_fix, (1000, 400))


# In[32]:


plt.imshow(img_new)


# ### Task 4: Resize & Flip an Image

# #### Resize with Ratio

# In[33]:


width_ratio = 0.5
height_ratio = 0.5


# In[34]:


img2 = cv2.resize(img_fix, (0,0), img_fix, width_ratio, height_ratio)


# In[35]:


plt.imshow(img2)


# In[36]:


img2.shape


# #### Flip on Horizontal Axis

# In[37]:


img_3 = cv2.flip(img_fix, 0)
plt.imshow(img_3)


# #### Flip on Vertical Axis

# In[38]:


img_3 = cv2.flip(img_fix, 1)
plt.imshow(img_3)


# #### Flip on Horizontal and Vertical Axis

# In[39]:


img_3 = cv2.flip(img_fix, -1)
plt.imshow(img_3)


# #### Change the size of our canva

# In[40]:


last_img = plt.figure(figsize=(10, 7))
ilp = last_img.add_subplot(111)
ilp.imshow(img_fix)


# ### Task 5: Draw Shapes on an Image (Part 1)

# #### Create a black image to work

# In[41]:


black_img = np.zeros(shape=(512,512,3), dtype=np.int16)


# #### Get the shape of the image

# In[42]:


black_img.shape


# #### Show it

# In[43]:


plt.imshow(black_img)


# #### Draw a Circle
# The center is the first number on x_axis and second on y-axis

# In[44]:


cv2.circle(img=black_img, center=(400,100), radius=50, color=(255,0,0), thickness=8)

plt.imshow(black_img)


# #### Filled Circle

# In[45]:


cv2.circle(img=black_img, center=(400,200), radius=50, color=(0,255,0), thickness=-1)

plt.imshow(black_img)


# #### Draw a Rectangle
# The first number is on x-axis and the second on the y-axis
# 
# We need two points. One for up and one diagonally down

# In[46]:


cv2.rectangle(black_img, pt1=(200,200), pt2=(300,300), color=(0,255,0), thickness=5)

plt.imshow(black_img)


# #### Draw Triangle

# In[47]:


vertices = np.array([[10,450], 
                    [110,350], 
                    [180,450]], 
                    np.int32)

pts = vertices.reshape(-1,1,2)

cv2.polylines(black_img, [pts], isClosed=True, color=(0,0,255), thickness=3)

plt.imshow(black_img)


# In[48]:


vertices = np.array([[10,450], 
                    [110,350], 
                    [180,450]], 
                    np.int32)


# In[49]:


vertices.shape


# In[50]:


pts = vertices.reshape(3,1,2)


# In[51]:


pts.shape


# In[52]:


cv2.polylines(black_img, [pts], isClosed=True, color=(0,0,255), thickness=3)

plt.imshow(black_img)


# #### Filled Rectangle

# In[53]:


cv2.rectangle(black_img, pt1=(350,450), pt2=(450,350), color=(0,255,0), thickness=-1)

plt.imshow(black_img)


# #### Filled Triangle

# In[54]:


vertices = np.array([[10,170], 
                    [110,50], 
                    [180,170]], 
                    np.int32)

pts = vertices.reshape(-1,1,2)

cv2.fillPoly(black_img, [pts], color=(0,0,255))

plt.imshow(black_img)


# #### Draw Line

# In[55]:


cv2.line(black_img, pt1=(512,0), pt2=(0,512), color=(255,0,255), thickness=3)

plt.imshow(black_img)


# #### Write Text

# In[56]:


font = cv2.FONT_HERSHEY_SIMPLEX

cv2.putText(black_img, text = 'OpenCV', org=(210,500), fontFace=font, 
            fontScale=1, color=(255,255,0), thickness=2, lineType=cv2.LINE_AA)

plt.imshow(black_img)


# ### Task 6: Draw with the Mouse

# In[57]:


# Function
# x,y, flags, param are feed from OpenCV automaticaly
def draw_circle(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img, (x,y), 70, (35,69,78), -1)
        
# Connect the Function with the Callback   
cv2.namedWindow(winname='my_drawing')

#Callback
cv2.setMouseCallback('my_drawing', draw_circle)

# Using OpenCV to show the Image
img = np.zeros((512,512,3), np.int8)

while True:
    cv2.imshow('my_drawing', img)
    
    if cv2.waitKey(5) & 0xFF == 27:
        break
        
cv2.destroyAllWindows()


# ### Task 7: Event Choices for the Mouse

# In[58]:


# Function
# x,y, flags, param are feed from OpenCV automaticaly
def draw_circle(event, x, y, flags, param):
    #Left Button Down
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img, (x,y), 100, (75,131,251), -1)
    #Right Button Down
    elif event == cv2.EVENT_RBUTTONDOWN:
        cv2.circle(img, (x,y), 50, (251,75,131), -1)
        
# Connect the Function with the Callback   
cv2.namedWindow(winname='my_draw')

#Callback
cv2.setMouseCallback('my_draw', draw_circle)

# Using OpenCV to show the Image
img = np.zeros((512,512,3), np.int8)

while True:
    cv2.imshow('my_draw', img)
    
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
        
cv2.destroyAllWindows()


# ### Task 8: Mouse Functionality

# In[59]:


# variables
# TRUE when the mouse botton is DOWN
# FALSE when the mouse button is UP
drawing = False
ex = -1
ey = -1

# Function
# x,y, flags, param are feed from OpenCV automaticaly
def draw_rectangle(event, x, y, flags, param):
    
    global ex, ey, drawing
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ex, ey, = x, y
    
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.rectangle(img, (ex,ey), (x,y),(255,0,255), -1)
            
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(img, (ex,ey), (x,y),(255,0,255), -1)
        
# Connect the Function with the Callback   
img = np.zeros((512,512,3),np.int8)

cv2.namedWindow(winname='my_draw')

#Callback
cv2.setMouseCallback('my_draw', draw_rectangle)

# Using OpenCV to show the Image
while True:
    cv2.imshow('my_draw', img)
    
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
        
cv2.destroyAllWindows()

