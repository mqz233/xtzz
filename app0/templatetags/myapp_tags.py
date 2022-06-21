from django import template
from django.utils.safestring import mark_safe
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
    elif(attr=="stage"):
        if(obj.get(attr) == "A"):
            strs = """
 <div class="radio">
                                  <label>
                                    <input type="radio" name="stage{0}"  value="A" checked form="choose">
                                    A
                                  </label>
                              </div>
                                <div class="radio">
                                  <label>
                                    <input type="radio" name="stage{0}"  value="B" form="choose">
                                    B
                                  </label>
                                </div>
                                                                <div class="radio">
                                  <label>
                                    <input type="radio" name="stage{0}"  value="C" form="choose">
                                    C
                                  </label>
                                </div>
                                                                <div class="radio">
                                  <label>
                                    <input type="radio" name="stage{0}"  value="D" form="choose">
                                    D
                                  </label>
                                </div>
        """.format(obj.get("frameId"))
        elif(obj.get(attr) == "B"):
            strs = """
             <div class="radio">
                                              <label>
                                                <input type="radio" name="stage{0}"  value="A" form="choose">
                                                A
                                              </label>
                                          </div>
                                            <div class="radio">
                                              <label>
                                                <input type="radio" name="stage{0}"  value="B" checked form="choose">
                                                B
                                              </label>
                                            </div>
                                                                            <div class="radio">
                                              <label>
                                                <input type="radio" name="stage{0}"  value="C" form="choose">
                                                C
                                              </label>
                                            </div>
                                                                            <div class="radio">
                                              <label>
                                                <input type="radio" name="stage{0}"  value="D" form="choose">
                                                D
                                              </label>
                                            </div>
                    """.format(obj.get("frameId"))
        elif(obj.get(attr) == "C"):
            strs = """
             <div class="radio">
                                              <label>
                                                <input type="radio" name="stage{0}"  value="A" form="choose">
                                                A
                                              </label>
                                          </div>
                                            <div class="radio">
                                              <label>
                                                <input type="radio" name="stage{0}"  value="B" form="choose">
                                                B
                                              </label>
                                            </div>
                                                                            <div class="radio">
                                              <label>
                                                <input type="radio" name="stage{0}"  value="C" checked form="choose">  
                                                C
                                              </label>
                                            </div>
                                                                            <div class="radio">
                                              <label>
                                                <input type="radio" name="stage{0}"  value="D" form="choose">
                                                D
                                              </label>
                                            </div>
                    """.format(obj.get("frameId"))
        else:
            strs = """
             <div class="radio">
                                              <label>
                                                <input type="radio" name="stage{0}"  value="A" form="choose">
                                                A
                                              </label>
                                          </div>
                                            <div class="radio">
                                              <label>
                                                <input type="radio" name="stage{0}"  value="B" form="choose">
                                                B
                                              </label>
                                            </div>
                                                                            <div class="radio">
                                              <label>
                                                <input type="radio" name="stage{0}"  value="C" form="choose">
                                                C
                                              </label>
                                            </div>
                                                                            <div class="radio">
                                              <label>
                                                <input type="radio" name="stage{0}"  value="D" checked form="choose">
                                                D
                                              </label>
                                            </div>
                    """.format(obj.get("frameId"))

        page_str_list = []
        page_str_list.append(strs)

        page_string = mark_safe("".join(page_str_list))
        return page_string
    else:
        if (obj.get(attr) == "good"):
            strs = """
         <div class="radio">
                                          <label>
                                            <input type="radio" name="eval{0}"  value="good" checked form="choose">
                                            good
                                          </label>
                                      </div>
                                        <div class="radio">
                                          <label>
                                            <input type="radio" name="eval{0}"  value="bad" form="choose">
                                            bad
                                          </label>
                                        </div>
                                                                        <div class="radio">
                                          <label>
                                            <input type="radio" name="eval{0}"  value="unknown" form="choose">
                                            unknown
                                          </label>
                                        </div>
                """.format(obj.get("frameId"))
        elif (obj.get(attr) == "bad"):
            strs = """
            <div class="radio">
                                             <label>
                                               <input type="radio" name="eval{0}"  value="good"  form="choose">
                                               good
                                             </label>
                                         </div>
                                           <div class="radio">
                                             <label>
                                               <input type="radio" name="eval{0}"  value="bad" form="choose" checked>
                                               bad
                                             </label>
                                           </div>
                                                                           <div class="radio">
                                             <label>
                                               <input type="radio" name="eval{0}"  value="unknown" form="choose">
                                               unknown
                                             </label>
                                           </div>
                   """.format(obj.get("frameId"))
        else:
            strs = """
            <div class="radio">
                                             <label>
                                               <input type="radio" name="eval{0}"  value="good"  form="choose">
                                               good
                                             </label>
                                         </div>
                                           <div class="radio">
                                             <label>
                                               <input type="radio" name="eval{0}"  value="bad" form="choose" >
                                               bad
                                             </label>
                                           </div>
                                                                           <div class="radio">
                                             <label>
                                               <input type="radio" name="eval{0}"  value="unknown" form="choose" checked>
                                               unknown
                                             </label>
                                           </div>
                   """.format(obj.get("frameId"))

        page_str_list = []
        page_str_list.append(strs)

        page_string = mark_safe("".join(page_str_list))
        return page_string

    # return obj.get(attr)

@register.filter
def deal_tag(obj):

    measurement = obj[5:]
    return measurement

# @register.filter
# def deal_key(obj):
#
#     measurement = ""
#     measurement = "ÉçÈº"+str(obj)
#     return measurement