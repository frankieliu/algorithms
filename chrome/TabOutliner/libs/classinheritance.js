/** @preserve Copyright 2012, 2013, 2014, 2015 by Vladyslav Volovyk. All Rights Reserved. */

/* Simple JavaScript Inheritance
 * By John Resig http://ejohn.org/
 * MIT Licensed.
 */
// Inspired by base2 and Prototype

// Example of usage http://ejohn.org/blog/simple-javascript-inheritance
//    var Person = Class.extend({
//      init: function(isDancing){
//        this.dancing = isDancing;
//      },
//      dance: function(){
//        return this.dancing;
//      }
//    });
//    var Ninja = Person.extend({
//      init: function(){
//        this._super( false );
//      },
//      dance: function(){
//        // Call the inherited version of dance()
//        return this._super();
//      },
//      swingSword: function(){
//        return true;
//      }
//    });
//
//    var p = new Person(true);
//    p.dance(); // => true
//
//    var n = new Ninja();
//    n.dance(); // => false
//    n.swingSword(); // => true
//
//    // Should all be true
//    p instanceof Person && p instanceof Class &&
//    n instanceof Ninja && n instanceof Person && n instanceof Class
//RR(function(){
  var initializing = false, fnTest = /qwerty/.test(function(){qwerty;}) ? /\b_super\b/ : /.*/;
  // The base Class implementation (does nothing)

  //RRwindow['Class'] = function(){};
  let Class = function(){};

  // Create a new Class that inherits from this class
  Class.extend = function(prop) {
    var _super = this.prototype;

    // Instantiate a base class (but only create the instance,
    // don't run the init constructor)
    initializing = true;
    var prototype = new this();
    initializing = false;

    function makeWrapperFunctionToEnableSuper(name, fn) {
        return function() {
            var tmp = this._super;

            // Add a new ._super() method that is the same method
            // but on the super-class
            this._super = _super[name];

            // The method only need to be bound temporarily, so we
            // remove it when we're done executing
            var ret = fn.apply(this, arguments);
            this._super = tmp;

            return ret;
        };
    }

    // Copy the properties over onto the new prototype
    for (var name in prop)
    {
        // Check if we're overwriting an existing function
        prototype[name] = ( typeof prop[name] == "function" && typeof _super[name] == "function" && fnTest.test(prop[name]) ) ?
                            makeWrapperFunctionToEnableSuper(name, prop[name])
                            :
                            prop[name];
    }

    // The dummy class constructor
    function Class() {
      // All construction is actually done in the init method
      if ( !initializing && this.init )
        this.init.apply(this, arguments);
    }

    // Populate our constructed prototype object
    Class.prototype = prototype;

    // Enforce the constructor to be what we expect
    Class.prototype.constructor = Class;

    // And make this class extendable
    Class.extend = arguments.callee;

    return Class;
  };
//RR})();