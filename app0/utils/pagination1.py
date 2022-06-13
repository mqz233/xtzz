"""
自定义的分页组件————想要使用这个组件需要做到：

在视图函数中：
    def pretty_list(request):

        # 1.根据自己的情况去筛选自己的数据
        queryset = models.PrettyNum.objects.all()

        # 2.实例化分页对象
        page_object = Pagination(request, queryset)

        context = {
            "queryset":page_object.page_queryset,   # 分完页的数据
            "page_string":page_object.html()   # 生成页码
            }

        return render(request, "pretty_list.html", context)

在HTML页面中
    {% for obj in queryset %}
        {{ obj.xx }}
    {% endfor %}

    <ul class="pagination"">
        {{ page_string }}
    </ul>

参数：
    request：请求的对象
    queryset：数据库中筛选出的符合条件的数据，要进行分页处理的数据
    page_size：每页展示多少条数据
    page_param：在URL中传递的获取分页的参数，例如：/pretty/list/?page=7
    plus：显示当前页的前、后xx页的按钮

"""
import influxdb
from django.utils.safestring import mark_safe


class Pagination1(object):

    def __init__(self, request, queryset, page_size=10, page_param="page", plus=5):

        from django.http.request import QueryDict
        import copy
        query_dict = copy.deepcopy(request.POST)
        query_dict._mutable = True
        self.query_dict = query_dict
        self.page_param = page_param

        page = request.POST.get(page_param, "1")
        if page.isdecimal():  # 如果输入的页码不是数字，那就默认为1
            page = int(page)
        else:
            page = 1
        self.page = page
        self.page_size = page_size

        self.start = (page - 1) * page_size
        self.end = page * page_size

        self.page_queryset = queryset[self.start:self.end]

        # 数据总条数
        if(isinstance(queryset,list)):
            total_count = len(queryset)
        else:
            total_count = queryset.count()
            # total_count = len(queryset)
        # 总页码
        total_page_count, div = divmod(total_count, page_size)
        if div:
            total_page_count += 1
        self.total_page_count = total_page_count
        self.plus = plus

    def html(self):
        # 计算出当前页的前plus页和后plus页
        if self.total_page_count <= 2 * self.plus + 1:
            start_page = 1
            end_page = self.total_page_count
        else:
            if self.page <= self.plus:
                start_page = 1
                end_page = 2 * self.plus + 1
            else:
                if (self.page + self.plus) > self.total_page_count:
                    start_page = self.total_page_count - 2 * self.plus
                    end_page = self.total_page_count
                else:
                    start_page = self.page - self.plus
                    end_page = self.page + self.plus

                    ## 一、页码按钮 由后台计算并加入到前端中
        page_str_list = []
        # 首页
        self.query_dict.setlist(self.page_param, [1])  # 在网址中保留原来的所有参数，再加入page参数
        # page_str_list.append('<li><a href="?{}">首页</a></li>'.format(self.query_dict.urlencode()))
        page_str_list.append('<li><a href="javascript: show({});">首页</a></li>'.format(self.query_dict.urlencode()))
        # 上一页
        if self.page > 1:
            self.query_dict.setlist(self.page_param, [self.page - 1])
            prev = '<li><a href="javascript: show({});">上一页</a></li>'.format(self.query_dict.urlencode())
        else:
            self.query_dict.setlist(self.page_param, [1])
            prev = '<li><a href="javascript: show({});">上一页</a></li>'.format(self.query_dict.urlencode())
        page_str_list.append(prev)
        # 页码
        for i in range(start_page, end_page + 1):
            self.query_dict.setlist(self.page_param, [i])
            if i == self.page:
                ele = '<li class="active"><a href="javascript: show({});">{}</a></li>'.format(self.query_dict.urlencode(), i)
            else:
                ele = '<li><a href="javascript: show({});">{}</a></li>'.format(self.query_dict.urlencode(), i)
            page_str_list.append(ele)
        # 下一页
        if self.page < end_page:
            # print(self.query_dict.urlencode())
            self.query_dict.setlist(self.page_param, [self.page + 1])
            prev = '<li><a href="javascript: show({});">下一页</a></li>'.format(self.query_dict.urlencode())
        else:
            self.query_dict.setlist(self.page_param, [self.total_page_count])
            prev = '<li><a href="javascript: show({});">下一页</a></li>'.format(self.query_dict.urlencode())
        page_str_list.append(prev)
        # 尾页
        self.query_dict.setlist(self.page_param, [self.total_page_count])
        page_str_list.append('<li><a href="javascript: show({});">尾页</a></li>'.format(self.query_dict.urlencode()))

        ## 二、页码搜索框
        search_string = """
            <li>
                <form style="float: left;margin-left: -1px;" method="get">
                    <input type="text" name="page" class="form-control" placeholder="页码"
                    style="position:relative;float:left;display:inline-block;width: 80px;border-radius:0;" >
                    <button style="border-radius: 0;" class="btn btn-default" type="submit">跳转</button>
                </form>
            </li>
        """
        page_str_list.append(search_string)

        page_string = mark_safe("".join(page_str_list))

        return page_string

