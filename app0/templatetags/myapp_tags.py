from django import template

register = template.Library()


@register.filter
def get_obj_attr(obj, attr):
    return getattr(obj, attr)


@register.filter
def get_dict_attr(obj, attr):
    return obj.get(attr)

@register.filter
def get_dict_attr2(obj, attr):
    if(attr=="frameId"):
        return obj.get(attr)+"-"+str(int(obj.get(attr))+49)

    return obj.get(attr)

@register.filter
def deal_tag(obj):

    measurement = obj[5:]
    return measurement