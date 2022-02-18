from git import Repo


def update_os_climate(repos_list, branches_list):
    for repository in repos_list:
        repo = Repo(repository)
        git = repo.git
        for branch in branches_list:
            git.checkout(branch)
            git.pull('origin', branch)
#            try:
#                git.pull('osclimate', branch, '--allow-unrelated-histories')
#            except:
#                print("upstream branch not present in osclimate, branch created")
            try:
                print(git.push('osclimate', branch))
            except:
                msg = "Something went wrong during the merge of branch %s "%branch
                msg += "of repository %s" %repository
                print(msg)
                raise


if '__main__' == __name__:
    update_os_climate([r'C:\Users\SP005188\eclipse-workspace\SoSTrades\sostrades-core'], ['develop'])
    print("update done")