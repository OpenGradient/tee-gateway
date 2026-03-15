import sys

if sys.version_info < (3, 7):
    import typing

    def is_generic(klass):
        """Determine whether klass is a generic class"""
        return type(klass) is typing.GenericMeta

    def is_dict(klass):
        """Determine whether klass is a Dict"""
        return klass.__extra__ is dict

    def is_list(klass):
        """Determine whether klass is a List"""
        return klass.__extra__ is list

else:

    def is_generic(klass):
        """Determine whether klass is a generic class"""
        return hasattr(klass, "__origin__")

    def is_dict(klass):
        """Determine whether klass is a Dict"""
        return klass.__origin__ is dict

    def is_list(klass):
        """Determine whether klass is a List"""
        return klass.__origin__ is list
