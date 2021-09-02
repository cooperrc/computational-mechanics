# Working in your ![fork image](../images/fork.png) Fork on GitHub.com

This is a short document on keeping track of your work on GitHub. There
five steps:
1. fork the
[computational-mechanics](https://github.com/cooperrc/computational-mechanics)
repository
2. clone your forked repository to your Jupyter server
3. keep your fork updated with a 'fetch upstream' on github.com
4. do your work then commit your changes
5. push your changes to github.com

__Steps 1 and 2 only have to be performed once.__

__Only do steps 3, 4, and 5 to keep your work up-to-date.__

## 1. fork the [computational-mechanics](https://github.com/cooperrc/computational-mechanics) repository
1. go to https://github.com/cooperrc/computational-mechanics
2. log in to github.com with your GitHub user name
2. click "fork" in the upper-right corner of the website

## 2. clone your forked repository to your Jupyter server
1. go to https://compmech.uconn.edu 
2. log in with your netid and password
3. Click "New" -> "Terminal"
4. type these commands, but replace `your-user-name` with your GitHub user name
```
cd work
git clone https://github.com/your-user-name/computational-mechanics.git
```
You now have a copy of your fork on your Jupyter server

## 3. keep your fork updated with a 'fetch upstream' on github.com
When changes are made to the main code, you'll want to update your files
in your fork. Do this in 2 steps:
1. go to `https://github.com/your-user-name/computational-mechanics`
replacing `your-user-name` with your GitHub user name
2. Click the button "Fetch upstream" ![fetch upstream button](../images/fetch-upstream.png)

## 4. do your work then commit your changes
After you have made changes e.g. you finished the first notebook's
problems, you'll want to "commit" those changes to git. Then, push them
to github.com

1. Save any work with the ![save button](../images/save-icon.png)
3. On the main [compmech site](https://compmech.uconn.edu) Click "New" -> "Terminal"
3. type these commands
```
cd work/computational-mechanics
git pull origin master
git add .
git commit -m 'my latest work message'
```

## 5. push your changes to github.com
Then, to __push the changes to github.com__ type this command
```
git push origin master
```
