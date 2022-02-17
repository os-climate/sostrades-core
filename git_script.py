from git import Repo
if '__main__' == __name__:
    #repo = Repo(r'C:\Users\SP005188\eclipse-workspace\SoSTrades\witness-energy')
    repo = Repo()
    print(repo.working_dir)
    git = repo.git
    #print(git.status())
    #print(git.remote('-v'))
    print(git.pull())
    #print(git.push('os'))
    #print(git.status())
    #print(git.status('origin',C=r'C:\Users\SP005188\eclipse-workspace\SoSTrades\witness-energy' ))
    print(git.checkout('develop'))
    print(git.pull('osclimate','develop','--allow-unrelated-histories'))
    print(git.push('osclimate','develop'))

    #cloned_repo = Repo.clone_from('https://idas661.eu.airbus.corp/sostrades/sostrades-core.git',r'C:\Users\SP005188\eclipse-workspace\SoSTrades\sostrades-core\git_test')
    #assert cloned_repo.__class__ is Repo
    print('a simple change')


