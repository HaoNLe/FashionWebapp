# How to run code on VM

### 1) Log in to paperspace machine

![paperspace_login](https://i.gyazo.com/873efabc061a5b8738361ebb1894583e.png)

### 2) Run Jupyter Notebook

Note: Remember to <kbd>cd</kbd> into your project directory first!

Run this to start jupyter in your paperspace terminal

<kbd> jupyter notebook --no-browser --port=8889 --NotebookApp.allow_remote_access=True</kbd> 

![jup_image](https://i.gyazo.com/b7b706095b35b00eea12d05e23914cb4.png)

### 3) SSH and redirect localhost

This step SSH's into your paperspace machine and redirects your localhost:8888 to your paperspace's localhost:8889 (which is the remote access port for your paperspace jupyter notebook)

Open a terminal on your computer (such as gitbash) and run the following:

<kbd>ssh -N -L localhost:8888:localhost:8889 paperspace@your.public.ip.here</kbd>

After entering your paperspace machine password the cursor will hang like so

![ssh_image](https://i.gyazo.com/b8d01c23ca19598ece8266d9dff7416e.png)

### 4) Open jupyter in browser

Copy/Paste the URL from your paperspace terminal into your browser. Make sure to replace 8889 with 8888.

If you URL has `(P64.137.4  dfsd):8889` instead of `localhost` replace the parenthesis and everything inside with `localhost` before running in your browser

<kbd>http://localhost:8888/?token=asdf1234asdf1234asdf1234</kbd>