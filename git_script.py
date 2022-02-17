from git import Repo


def update_os_climate(repos_list, branches_list):
    for repository in repos_list:
        repo = Repo(repository)
        git = repo.git
        for branch in branches_list:
            git.pull('origin', branch)
            git.checkout(branch)
            try:
                git.pull('osclimate', branch, '--allow-unrelated-histories')
            except:
                print("upstream branch not present in osclimate, branch created")
            git.push('osclimate', branch)
    print("done 3 ")


if '__main__' == __name__:
    update_os_climate([r'C:\Users\SP005188\eclipse-workspace\SoSTrades\sostrades-core'], ['develop'])