def should_retry(error_name):
    permanent_errors = ['ValidationError']
    return error_name not in permanent_errors
