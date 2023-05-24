"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports["default"] = shallowEqual;
var _objectIs = _interopRequireDefault(require("object-is"));
var _has = _interopRequireDefault(require("has"));
function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { "default": obj }; }
function _typeof(obj) { "@babel/helpers - typeof"; return _typeof = "function" == typeof Symbol && "symbol" == typeof Symbol.iterator ? function (obj) { return typeof obj; } : function (obj) { return obj && "function" == typeof Symbol && obj.constructor === Symbol && obj !== Symbol.prototype ? "symbol" : typeof obj; }, _typeof(obj); }
// adapted from https://github.com/facebook/react/blob/144328fe81719e916b946e22660479e31561bb0b/packages/shared/shallowEqual.js#L36-L68
function shallowEqual(objA, objB) {
  if ((0, _objectIs["default"])(objA, objB)) {
    return true;
  }
  if (!objA || !objB || _typeof(objA) !== 'object' || _typeof(objB) !== 'object') {
    return false;
  }
  var keysA = Object.keys(objA);
  var keysB = Object.keys(objB);
  if (keysA.length !== keysB.length) {
    return false;
  }
  keysA.sort();
  keysB.sort();

  // Test for A's keys different from B.
  for (var i = 0; i < keysA.length; i += 1) {
    if (!(0, _has["default"])(objB, keysA[i]) || !(0, _objectIs["default"])(objA[keysA[i]], objB[keysA[i]])) {
      return false;
    }
  }
  return true;
}
module.exports = exports.default;
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJuYW1lcyI6WyJzaGFsbG93RXF1YWwiLCJvYmpBIiwib2JqQiIsImlzIiwia2V5c0EiLCJPYmplY3QiLCJrZXlzIiwia2V5c0IiLCJsZW5ndGgiLCJzb3J0IiwiaSIsImhhcyJdLCJzb3VyY2VzIjpbIi4uL3NyYy9pbmRleC5qcyJdLCJzb3VyY2VzQ29udGVudCI6WyJpbXBvcnQgaXMgZnJvbSAnb2JqZWN0LWlzJztcbmltcG9ydCBoYXMgZnJvbSAnaGFzJztcblxuLy8gYWRhcHRlZCBmcm9tIGh0dHBzOi8vZ2l0aHViLmNvbS9mYWNlYm9vay9yZWFjdC9ibG9iLzE0NDMyOGZlODE3MTllOTE2Yjk0NmUyMjY2MDQ3OWUzMTU2MWJiMGIvcGFja2FnZXMvc2hhcmVkL3NoYWxsb3dFcXVhbC5qcyNMMzYtTDY4XG5leHBvcnQgZGVmYXVsdCBmdW5jdGlvbiBzaGFsbG93RXF1YWwob2JqQSwgb2JqQikge1xuICBpZiAoaXMob2JqQSwgb2JqQikpIHtcbiAgICByZXR1cm4gdHJ1ZTtcbiAgfVxuXG4gIGlmICghb2JqQSB8fCAhb2JqQiB8fCB0eXBlb2Ygb2JqQSAhPT0gJ29iamVjdCcgfHwgdHlwZW9mIG9iakIgIT09ICdvYmplY3QnKSB7XG4gICAgcmV0dXJuIGZhbHNlO1xuICB9XG5cbiAgY29uc3Qga2V5c0EgPSBPYmplY3Qua2V5cyhvYmpBKTtcbiAgY29uc3Qga2V5c0IgPSBPYmplY3Qua2V5cyhvYmpCKTtcblxuICBpZiAoa2V5c0EubGVuZ3RoICE9PSBrZXlzQi5sZW5ndGgpIHtcbiAgICByZXR1cm4gZmFsc2U7XG4gIH1cblxuICBrZXlzQS5zb3J0KCk7XG4gIGtleXNCLnNvcnQoKTtcblxuICAvLyBUZXN0IGZvciBBJ3Mga2V5cyBkaWZmZXJlbnQgZnJvbSBCLlxuICBmb3IgKGxldCBpID0gMDsgaSA8IGtleXNBLmxlbmd0aDsgaSArPSAxKSB7XG4gICAgaWYgKCFoYXMob2JqQiwga2V5c0FbaV0pIHx8ICFpcyhvYmpBW2tleXNBW2ldXSwgb2JqQltrZXlzQVtpXV0pKSB7XG4gICAgICByZXR1cm4gZmFsc2U7XG4gICAgfVxuICB9XG5cbiAgcmV0dXJuIHRydWU7XG59XG4iXSwibWFwcGluZ3MiOiI7Ozs7OztBQUFBO0FBQ0E7QUFBc0I7QUFBQTtBQUV0QjtBQUNlLFNBQVNBLFlBQVksQ0FBQ0MsSUFBSSxFQUFFQyxJQUFJLEVBQUU7RUFDL0MsSUFBSSxJQUFBQyxvQkFBRSxFQUFDRixJQUFJLEVBQUVDLElBQUksQ0FBQyxFQUFFO0lBQ2xCLE9BQU8sSUFBSTtFQUNiO0VBRUEsSUFBSSxDQUFDRCxJQUFJLElBQUksQ0FBQ0MsSUFBSSxJQUFJLFFBQU9ELElBQUksTUFBSyxRQUFRLElBQUksUUFBT0MsSUFBSSxNQUFLLFFBQVEsRUFBRTtJQUMxRSxPQUFPLEtBQUs7RUFDZDtFQUVBLElBQU1FLEtBQUssR0FBR0MsTUFBTSxDQUFDQyxJQUFJLENBQUNMLElBQUksQ0FBQztFQUMvQixJQUFNTSxLQUFLLEdBQUdGLE1BQU0sQ0FBQ0MsSUFBSSxDQUFDSixJQUFJLENBQUM7RUFFL0IsSUFBSUUsS0FBSyxDQUFDSSxNQUFNLEtBQUtELEtBQUssQ0FBQ0MsTUFBTSxFQUFFO0lBQ2pDLE9BQU8sS0FBSztFQUNkO0VBRUFKLEtBQUssQ0FBQ0ssSUFBSSxFQUFFO0VBQ1pGLEtBQUssQ0FBQ0UsSUFBSSxFQUFFOztFQUVaO0VBQ0EsS0FBSyxJQUFJQyxDQUFDLEdBQUcsQ0FBQyxFQUFFQSxDQUFDLEdBQUdOLEtBQUssQ0FBQ0ksTUFBTSxFQUFFRSxDQUFDLElBQUksQ0FBQyxFQUFFO0lBQ3hDLElBQUksQ0FBQyxJQUFBQyxlQUFHLEVBQUNULElBQUksRUFBRUUsS0FBSyxDQUFDTSxDQUFDLENBQUMsQ0FBQyxJQUFJLENBQUMsSUFBQVAsb0JBQUUsRUFBQ0YsSUFBSSxDQUFDRyxLQUFLLENBQUNNLENBQUMsQ0FBQyxDQUFDLEVBQUVSLElBQUksQ0FBQ0UsS0FBSyxDQUFDTSxDQUFDLENBQUMsQ0FBQyxDQUFDLEVBQUU7TUFDL0QsT0FBTyxLQUFLO0lBQ2Q7RUFDRjtFQUVBLE9BQU8sSUFBSTtBQUNiO0FBQUMifQ==
//# sourceMappingURL=index.js.map