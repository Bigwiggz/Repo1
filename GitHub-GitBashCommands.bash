

########################################
Git and Git Hub
########################################


1) git --version

#Gives you the version of git on your local machine

2) git init

#initialize git for a project within the folder you are init

3) git clone

ex: git clone git@github.com:gwenf/demo-repo.git@github

#Bring a repository that is hosted somewhere like Github into a folder on your local machine.

4) git add 

ex: git add . OR "filename"

#this adds all files "." or adds "filename" to staging area to track changes in files

5) git commit

ex: git commit -m "insert commit name here" -m "insert description here"

#Saves your files in Git

6) git status

#This shows all the files that are being tracked for changes and staged for commit



########################################
#GitHub Repo Only
########################################

6) git push

ex: git push origin master OR git push -u

#Upload git commits to a remote repository, like Github.  (The example pushes to the remote repo master branch.)  -u can replace origin master.

7) git pull

ex: git pull ---

#Download changes from your remote repositoryto our local machine, the opposite of push.

8) git remote add origin

#This command adds a GitHub repo that is not on your computer as an origin connection.  You only have to do this once per coding session.

9) git remote -v

#This command shows the remote repositories connected to your local machine.

10) 


########################################
#How to start a GitHub account
########################################

1) Go to Github.com create an account with a username and password.

2) You can go to "New" to create a new repository. Or you can do it on the CLI as shown below

2a) Create a README.md to explain the purpose of your github repository

3) Create new repository using Github CLI

3a) echo"#demo-repo">> README.md
	git init
	git add README.md
	git commit -m "first commit"
	git remote add origin git@github.com:gwenf/demo-repo.git@github
	git push -u origin master
	

########################################
#Bash Terminal Commands
########################################

ls -la
Show all folders including hidden folders

cd
Enter into directory

########################################
#GitHub generate SSH keys to link to accounts
########################################

1) enter in the following command below

ssh-keygen -t -rsa -b 4096 -C "enteremail@email.com"

2) When prompted enter file in which to save the keygen

3) When prompted enter password

4) you can use ls | grep testkey to search for the generated key and key.pub (key is your private key on your local machine while key.pub is your public key that goes on Github.)

5) Print out public key using "cat key.pub"

6) In the bash terminal just highlighting the key will copy it to clipboard.  If that doesnt work you can use the following command.  pbcopy <~/testkey.pub

7) Go to your github repo and click SSH keys. -> New SSH key

8)  Then you need to start ssh-agent in the background and follow the instructions to add an ssh key in GitHub help


########################################
#Git Branching
########################################


1) git branch

#This tells you which branch you are on

2) git checkout-b fbranch-name

#This is to switch to a branch with the name "branch-name"

3) git diff

#This shows you all the lines that have changed between versions

4) git merge branch-name

#This command merges "branch-name" with the master branch

5) git push

#In conjunction with branching, this pushes any changes to the repository.
#Then copy and paste the link shown below the command as a git push

git push --set-upstream origin branch-name

6) Then you can go to your GitHub account and confirm pull request by commenting on it and do it.

Then you can use git pull to pull down the changes you made on your local machine.

########################################
#How to handle merge conflicts
########################################

Use VS code to sort out changes then save changes

########################################
#Undo Git
########################################

1) git reset HEAD~1

#Rollback 1 committ to unstage and rollback committe

2) git log

#display log of all committ in reverse chronilogical order.

3) git reset HASH

#get go back to a specific committ

4) git reset --hard HASH

#Go back to a specific committ and delete all changes made after that committ

########################################
#Forking a Repo
########################################

1) Forking can be done on GitHub
