import datetime

def format_timedelta(td: datetime.timedelta, how: str='hh:mm:ss') -> str:
	"""
	Formats a time delta as hr:min:secs for logging purposes

	Parameters
	----------
	td: datetime.timedelta
		the timedelta we wish to format

	how: str (default = 'hh:mm:ss')
		the method we wish to use to format our timedelta
		only two methods implemented so far: hh:mm:ss and total seconds elapsed
	"""
	total_seconds = td.total_seconds()
	hours = total_seconds // 3600
	mins = (total_seconds % 3600) // 60
	secs = (total_seconds % 3600) % 60
	if how == 'hh:mm:ss':
		return '%d:%d:%d' %(hours, mins, secs)
	elif how == 'seconds':
		return total_seconds
	else:
		raise ValueError('Invalid format.')