# Getting Data Onto Your VM

In order to train our models we first need data.

There are many ways of obtaining said data. Examples include simply using <kbd>curl</kbd> or <kbd>wget</kbd> commands in your paperspace terminal to download that data directly into your machine.

Lets get data from kaggle

## Setting up Kaggle API access on VM 
<kbd></kbd>
1. Create a Kaggle account if you do not have one already
2. Go to https://www.kaggle.com/username/account by clicking on your icon and then my account
3. Scroll down to <kbd>API</kbd> (about midway) and then click <kbd>Create New API Token</kbd>
4. This downloads <kbd>kaggle.json</kbd>. Open it.
5. Login to your paperspace machine
6. Set your kaggle username in your Unix environment: <kbd>export KAGGLE_USERNAME=username</kbd>
7. Set your kaggle API key in your Unix environment: <kbd>export KAGGLE_KEY=xxxxxxxxxxxxxx</kbd>
8. Download mnist fashion files: <kbd>kaggle datasets download zalando-research/fashionmnist</kbd>
    * note: to download other datasets just replace `zalando-research/fashionmnist` with whatever `submitter/dataname` you want to download


## Moving and unzipping the .zip data
See bottom of the doc for a mini reference to some basic Unix commands.
1. Make a directory within project directory <kbd>mkdir FashionWebapp/fashiondata</kbd>
    * Note: I consider it good practice to create a separate folder for each data.zip to keep everything separated. Not only that, but when unzipping it can be unpredictable what comes out. Sometimes it can be just a bunch of images (which would've been bad had we not created a separate directory) or it could be already separated neatly into a training and test folder.
2. move the zip into your newly created directory <kbd>mv fashionmnist.zip FashionWebapp/fashiondata/fashionmnist.zip</kbd>
3. cd into said directory <kbd>cd FashionWebapp/fashiondata</kbd>
4. unzip the data <kbd>unzip fashionmnist.zip</kbd>
5. Running <kbd>ls</kbd> should show the files unzipped.
6. Now go ahead and delete the .zip file <kbd>rm fashionmnist.zip</kbd>

## Unix Commands
You'll be needing these in order to use your own datasets.

**Making a directory**

```
mkdir mydir                 # Creates a new directory
mkdir -m a=rwx mydir        # Creates a new directory and set permissions so all users may read, write, and execute the contents.
mkdir test1 test2 test3     # Creates multiple directories
mkdir -p /home/test/test1/test2/test3/test4
                            # Creates multiple subdirectory levels at once
```

**Moving files/directories**
```
mv myfile mynewfilename     # renames 'myfile' to 'mynewfilename'.
mv myfile ~/myfile          # moves 'myfile' from the current directory to user's home directory.
                            # the notation '~' refers to the user's "home" (login) directory
mv myfile subdir/myfile     # moves 'myfile' to 'subdir/myfile' relative to the current directory.
mv myfile subdir            # same as the previous command, filename is implied to be the same.
mv myfile subdir/myfile2    # moves 'myfile' to 'subdir' named  'myfile2'.
mv be.03 /mnt/bkup/bes      # copies 'be.03' to the mounted volume 'bkup' the 'bes' directory, 
                            # then 'be.03' is removed.
mv afile another /home/yourdir/yourfile mydir 
                            # moves multiple files to directory 'mydir'.
mv /var/log/*z ~/logs       # takes longer than expected if '/var' is on a different file system, 
                            # as it frequently is, since files will be copied & deleted
                            # be careful when using globbable file name patterns containing
                            # the characters ?*[ to ensure that the arguments passed to 'mv'
                            # include a list of non-directories and a terminating directory

man mv                      # displays the complete UNIX manual page for the 'mv' command.
```

**Unzipping a zipped file**
```
unzip training_set.zip      # Unzips training_set in its current directory
```
You'll want to move the zipped file into its own folder before unzipping like so:
```
mv training_set.zip training_set/training_set.zip
cd training_set
unzip training_set.zip
```

**Deleting a file**
```
rm filename
```

**Deleting a directory along with files/directories inside it**
```
rm -r directoryname
```
