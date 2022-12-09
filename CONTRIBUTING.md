## Contributing

First off, thank you for considering contributing to kompil. It's people like you that will enable
kompil to be a useful codec.

### Where do I go from here?

If you've noticed a bug or have a feature request, [make one][new issue]! It's generally best if you
get confirmation of your bug or approval for your feature request this way before starting to code.

### Fork & create a branch

If this is something you think you can fix, then [fork kompil] and create a branch with a
descriptive name.

A good branch name would be (where issue #145 is the ticket you're working on):

```sh
git checkout -b 145-remove-unused-lines
```

### Implement your fix or feature

At this point, you're ready to make your changes! Feel free to ask for help; everyone is a beginner
at first :smile_cat:

### Get the style right

Your patch should follow the same conventions & pass the same code quality checks as the rest of the
project.

You can run `black .` to format the python code. This is mandatory for pull requests to be accepted.

### Make a Pull Request

At this point, you should switch back to your master branch and make sure it's up to date with
kompil's master branch:

```sh
git remote add upstream git@github.com:kompil/kompil.git
git checkout master
git pull upstream master
```

Then update your feature branch from your local copy of master, and push it!

```sh
git checkout 145-remove-unused-lines
git rebase master
git push --set-upstream origin 145-remove-unused-lines
```

Finally, go to GitHub and [make a Pull Request][] :D

### Keeping your Pull Request updated

If a maintainer asks you to "rebase" your PR, they're saying that a lot of code has changed, and
that you need to update your branch so it's easier to merge.

To learn more about rebasing in Git, there are a lot of [good][git rebasing]
[resources][interactive rebase] but here's the suggested workflow:

```sh
git checkout 145-remove-unused-lines
git pull --rebase upstream master
git push --force-with-lease 145-remove-unused-lines
```

[new issue]: https://github.com/kompil/kompil/issues/new
[fork kompil]: https://help.github.com/articles/fork-a-repo
[make a pull request]: https://help.github.com/articles/creating-a-pull-request
[git rebasing]: http://git-scm.com/book/en/Git-Branching-Rebasing
[interactive rebase]: https://help.github.com/en/github/using-git/about-git-rebase
[shortcut reference links]: https://github.github.com/gfm/#shortcut-reference-link
