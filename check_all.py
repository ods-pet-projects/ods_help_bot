import os
from config import ROOT_DIR, index_path


class ResourceChecker:
    def log(self, msg):
        print(msg)

    def check_all(self):
        check_dict = {'check_dirs': self.check_dirs,
                      'check_env': self.check_env,
                      'check_data': self.check_data,
                      'check_elastic': self.check_elastic,
                      }
        need_checks = ['check_dirs', 'check_env', 'check_data', 'check_elastic']
        failed_steps = []
        status = True
        for check in need_checks:
            cur_status = check_dict[check]()
            if not cur_status:
                failed_steps.append(check)
            status &= cur_status

        if status:
            self.log('check_all passed')
        else:
            self.log(f'check_all failed, faled steps: {failed_steps}')

    def check_dirs(self):
        status = os.path.exists(ROOT_DIR)
        true_msg = 'check_dirs passed'
        false_msg = f'check_dirs failed not found root dir {status}'
        self.log(true_msg if status else false_msg)
        return status

    def check_variable(self, var_name):
        if var_name in os.environ:
            self.log(f'\t{var_name} variable found')
            return True
        self.log(f'\t{var_name} variable not found')
        return False

    def check_env(self):
        status = True
        var_names = ['TOKEN']
        self.log('check_env')
        for var_name in var_names:
            status &= self.check_variable(var_name)
        self.log('\tpassed') if status else self.log('\tfailed')
        return status

    def check_data(self):
        status = True
        self.log('check_data')
        for model_name, path in index_path.items():
            if os.path.exists(path):
                self.log(f'\t{model_name} index found at {path}')
            else:
                self.log(f'\t{model_name} index not found at {path}')
                status = False
        self.log('\tpassed') if status else self.log('\t failed')
        return status

    def check_elastic(self):
        try:
            from ml_models.elastic_search_baseline import get_answer
            ans = get_answer('data science meetups')
            self.log('elastic passed')
            status = True
        except:
            status = False
            self.log('elastic failed')
        return status


def run_validation():
    print('run validation started')
    rc = ResourceChecker()
    rc.check_all()


if __name__ == '__main__':
    run_validation()
